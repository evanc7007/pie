//! OLMo 3 instruct implementation.
//!
//! Implements OLMo 3 chat template:
//! - ChatML-style: <|im_start|>role\ncontent<|im_end|>\n
//! - Tools defined in <functions>...</functions> within system/user messages.
//! - Tool calls in <function_calls>...</function_calls> within assistant messages.
//! - Tool outputs in <|im_start|>environment\ncontent<|im_end|>\n.
//! - Generation prompt adds <|im_start|>assistant\n<think>

use crate::inference::structured::grammar::Grammar;
use crate::model::instruct::decoders::{GenericChatDecoder, ThinkingDecoder};
use crate::model::instruct::{
    ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder, ToolEvent, ToolGrammar,
};
use crate::model::tokenizer::Tokenizer;
use std::sync::Arc;

static TEMPLATE: &str = r#"
{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}{% if not has_system %}{{ '<|im_start|>system
You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>
' }}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system
' + message['content'] }}{% if message.get('functions', none) is not none %}{{ ' <functions>' + message['functions'] + '</functions><|im_end|>
' }}{% else %}{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>
' }}{% endif %}{% elif message['role'] == 'user' %}{% if message.get('functions', none) is not none %}{{ '<|im_start|>user
' + message['content'] + '
' + '<functions>' + message['functions'] + '</functions><|im_end|>
' }}{% else %}{{ '<|im_start|>user
' + message['content'] + '<|im_end|>
' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant
' }}{% if message.get('content', none) is not none %}{{ message['content'] }}{% endif %}{% if message.get('function_calls', none) is not none %}{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}{% endif %}{% if not loop.last %}{{ '<|im_end|>' + '
' }}{% else %}{{ eos_token }}{% endif %}{% elif message['role'] == 'environment' %}{{ '<|im_start|>environment
' + message['content'] + '<|im_end|>
' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant
<think>' }}{% endif %}{% endfor %}"#;

pub struct OlmoInstruct {
    tokenizer: Arc<Tokenizer>,
    im_start: Vec<u32>,
    im_end: Vec<u32>,
    newline: Vec<u32>,
    eos_token: Vec<u32>,
    // Roles
    system_role: Vec<u32>,
    user_role: Vec<u32>,
    assistant_role: Vec<u32>,
    environment_role: Vec<u32>,
    // Tools
    functions_start: Vec<u32>,
    functions_end: Vec<u32>,
    fn_calls_start: Vec<u32>,
    fn_calls_end: Vec<u32>,
    // Generation
    think_start: Vec<u32>,
    think_end: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl OlmoInstruct {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let encode = |s: &str| tokenizer.encode(s);

        let im_start = encode("<|im_start|>");
        let im_end = encode("<|im_end|>");
        let newline = encode("\n");
        let eos_token = encode("<|endoftext|>");

        let mut stop_ids = im_end.clone();
        stop_ids.extend(&eos_token);

        Self {
            im_start,
            im_end,
            newline,
            eos_token,
            system_role: encode("system"),
            user_role: encode("user"),
            assistant_role: encode("assistant"),
            environment_role: encode("environment"),
            functions_start: encode("<functions>"),
            functions_end: encode("</functions>"),
            fn_calls_start: encode("<function_calls>"),
            fn_calls_end: encode("</function_calls>"),
            think_start: encode("<think>"),
            think_end: encode("</think>"),
            stop_ids,
            tokenizer,
        }
    }

    fn wrap(&self, role: &[u32], content: &str) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(role);
        tokens.extend(&self.newline);
        tokens.extend(self.tokenizer.encode(content));
        tokens.extend(&self.im_end);
        tokens.extend(&self.newline);
        tokens
    }

    /// Renders `name(key1=value1, key2=value2, ...)`, matching the
    /// reference jinja: `key ~ '=' ~ (value | tojson)`, args joined by
    /// `", "`. `serde_json::Value`'s `Display` impl is exactly `tojson`
    /// (compact JSON), so `{v}` below is the reference behavior.
    fn render_one_call(name: &str, arguments_json: &str) -> String {
        let args: serde_json::Value = serde_json::from_str(arguments_json)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        let parts: Vec<String> = args
            .as_object()
            .map(|obj| obj.iter().map(|(k, v)| format!("{k}={v}")).collect())
            .unwrap_or_default();
        format!("{name}({})", parts.join(", "))
    }

    /// Builds an EBNF grammar constraining generation to
    /// `<function_calls>name(key=json, ...)[\nname2(...)]*</function_calls>`.
    /// Mirrors `QwenInstruct`/`R1Instruct`'s `build_tool_call_grammar`: tool
    /// names come from each schema's `function.name`/`name`; arguments are
    /// constrained by the same generic JSON-value grammar those archs use.
    fn build_tool_call_grammar(tools: &[String]) -> Option<String> {
        let mut names: Vec<String> = Vec::new();
        for tool in tools {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(tool) {
                let name = parsed
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .or_else(|| parsed.get("name"))
                    .and_then(|n| n.as_str());
                if let Some(n) = name {
                    names.push(format!("\"{n}\""));
                }
            }
        }
        if names.is_empty() {
            return None;
        }
        let name_alt = names.join(" | ");
        Some(format!(
            r#"root ::= "<function_calls>" tool-call ("\n" tool-call)* "</function_calls>"
tool-call ::= tool-name "(" arguments? ")"
tool-name ::= {name_alt}
arguments ::= argument ("," " " argument)*
argument ::= arg-key "=" json-value
arg-key ::= [a-zA-Z_] [a-zA-Z0-9_]*
json-object ::= "{{" json-members? "}}"
json-members ::= json-pair ("," json-pair)*
json-pair ::= json-string ":" json-value
json-value ::= json-string | json-number | json-object | json-array | "true" | "false" | "null"
json-string ::= "\"" json-chars "\""
json-chars ::= json-char*
json-char ::= [^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-array ::= "[" (json-value ("," json-value)*)? "]"
"#
        ))
    }
}

impl Instruct for OlmoInstruct {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.system_role, msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.wrap(&self.user_role, msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(&self.assistant_role);
        tokens.extend(&self.newline);
        tokens.extend(self.tokenizer.encode(msg));
        tokens.extend(&self.im_end);
        tokens.extend(&self.newline);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        let mut tokens = self.im_start.clone();
        tokens.extend(&self.assistant_role);
        tokens.extend(&self.newline);
        tokens.extend(&self.think_start);
        tokens
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    fn equip(&self, tools: &[String]) -> Vec<u32> {
        if tools.is_empty() {
            return Vec::new();
        }
        let preamble = "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024. ";
        let mut msg = preamble.to_string();
        msg.push_str("<functions>");
        msg.push_str(&tools.join("\n"));
        msg.push_str("</functions>");

        self.system(&msg)
    }

    fn answer(&self, _name: &str, value: &str) -> Vec<u32> {
        self.wrap(&self.environment_role, value)
    }

    // Matches the reference chat template: all of a turn's calls are
    // wrapped in a single `<function_calls>...</function_calls>`, each
    // call rendered as `name(key=value, ...)` (`value` via `tojson`),
    // newline-joined — NOT the per-call-tagged JSON format other archs
    // (e.g. Qwen's `<tool_call>{...}</tool_call>`) use.
    fn render_tool_calls(&self, calls: &[(String, String)]) -> String {
        let inner = calls
            .iter()
            .map(|(name, arguments_json)| Self::render_one_call(name, arguments_json))
            .collect::<Vec<_>>()
            .join("\n");
        format!("<function_calls>{inner}</function_calls>")
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.im_end.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        // Starts inside because cue() includes <think>; empty start_ids = starts inside
        Box::new(ThinkingDecoder::new(
            self.tokenizer.clone(),
            vec![],
            self.think_end.clone(),
        ))
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(OlmoToolDecoder {
            tokenizer: self.tokenizer.clone(),
            accumulated: String::new(),
            state: ToolState::Outside,
            current_tag: String::new(),
        })
    }

    fn tool_call_grammar(&self, tools: &[String]) -> Option<ToolGrammar> {
        let source = Self::build_tool_call_grammar(tools)?;
        let grammar = Grammar::from_ebnf(&source, "root").ok()?;
        Some(ToolGrammar {
            source,
            grammar: Arc::new(grammar),
        })
    }
}

// ─── Decoders ───────────────────────────────────────────────

struct OlmoToolDecoder {
    tokenizer: Arc<Tokenizer>,
    accumulated: String,
    state: ToolState,
    current_tag: String,
}

#[derive(Debug, PartialEq)]
enum ToolState {
    Outside,
    Inside,
}

impl ToolDecoder for OlmoToolDecoder {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent {
        let text = self.tokenizer.decode(tokens, false);
        self.accumulated.push_str(&text);

        loop {
            match self.state {
                ToolState::Outside => {
                    if let Some(pos) = self.accumulated.find("<function_calls>") {
                        self.accumulated =
                            self.accumulated[pos + "<function_calls>".len()..].to_string();
                        self.state = ToolState::Inside;
                        continue;
                    }
                    if self.accumulated.len() > 200 {
                        let keep = self.accumulated.len() - 50;
                        self.accumulated = self.accumulated[keep..].to_string();
                    }
                    return ToolEvent::Start;
                }
                ToolState::Inside => {
                    if let Some(pos) = self.accumulated.find("</function_calls>") {
                        let content = self.accumulated[..pos].trim().to_string();
                        self.accumulated =
                            self.accumulated[pos + "</function_calls>".len()..].to_string();
                        self.state = ToolState::Outside;

                        // Native format is `name(key=value, ...)` call
                        // syntax (matching the reference chat template's
                        // replay rendering), not JSON — a group can
                        // contain multiple newline-joined calls.
                        // TODO(parallel-tool-calls): only the first call
                        // in a multi-call group is surfaced; the rest are
                        // dropped. `ToolEvent::Call` carries one call, and
                        // tau-bench-style single-call-per-turn usage is
                        // the common case this was scoped for.
                        if let Some((name, args)) = split_calls(&content)
                            .first()
                            .and_then(|first_call| parse_call_syntax(first_call))
                        {
                            return ToolEvent::Call(name, args);
                        }
                        return ToolEvent::Start;
                    }
                    return ToolEvent::Start;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.accumulated.clear();
        self.state = ToolState::Outside;
    }
}

/// Splits a `<function_calls>` body into individual `name(args)` calls.
/// Calls are newline-joined per the reference template; a literal `\n`
/// inside a call can only occur as this separator because JSON-encoded
/// argument values (`tojson`) never emit a raw newline byte, so
/// splitting on raw `\n` is safe.
fn split_calls(content: &str) -> Vec<&str> {
    content
        .split('\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parses one `name(key=value, ...)` call into `(name, arguments_json)`.
/// Values are JSON (`tojson` output) and may contain literal `(` `)` `,`
/// inside quoted strings, so this scans char-by-char tracking
/// string/bracket-depth state rather than splitting naively.
fn parse_call_syntax(call: &str) -> Option<(String, String)> {
    let open = call.find('(')?;
    let name = call[..open].trim().to_string();

    let bytes = call.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut close = None;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let args_str = &call[open + 1..close];

    let mut map = serde_json::Map::new();
    for piece in split_top_level_commas(args_str) {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let eq = piece.find('=')?;
        let key = piece[..eq].trim().to_string();
        let value_text = piece[eq + 1..].trim();
        let value = serde_json::from_str::<serde_json::Value>(value_text)
            .unwrap_or_else(|_| serde_json::Value::String(value_text.to_string()));
        map.insert(key, value);
    }
    Some((name, serde_json::Value::Object(map).to_string()))
}

/// Splits `args_str` on top-level commas — not inside a quoted string,
/// `[...]`, or `{...}` (argument values may be JSON arrays/objects).
fn split_top_level_commas(args_str: &str) -> Vec<&str> {
    let bytes = args_str.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' | b'{' => depth += 1,
            b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&args_str[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&args_str[start..]);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tokenizer::Tokenizer;
    use std::sync::Arc;

    fn make_tok(vocab: &[&str]) -> Arc<Tokenizer> {
        let v: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
        Arc::new(Tokenizer::from_vocab(&v))
    }

    #[test]
    fn system_format() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "system", "Hello"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.system("Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>system\nHello<|im_end|>\n"));
    }

    #[test]
    fn equip_format() {
        // Build the exact content string that equip() will encode so the
        // mock tokenizer's fast-path recognizes it as a single token.
        let tools = &["foo".to_string(), "bar".to_string()];
        let preamble = "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024. ";
        let content = format!("{}<functions>{}</functions>", preamble, tools.join("\n"));
        let mut vocab: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "\n",
            "system",
            "<functions>",
            "</functions>",
            "foo",
            "bar",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        vocab.push(content);
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.equip(tools);
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<functions>"));
        assert!(text.contains("foo"));
        assert!(text.contains("bar"));
        assert!(text.contains("</functions>"));
    }

    #[test]
    fn answer_format() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "environment", "result"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.answer("fn", "result");
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>environment\nresult<|im_end|>\n"));
    }

    #[test]
    fn generation_cue_includes_think() {
        let tok = make_tok(&["<|im_start|>", "<|im_end|>", "\n", "assistant", "<think>"]);
        let inst = OlmoInstruct::new(tok);
        let tokens = inst.cue();
        let text = inst.tokenizer.decode(&tokens, false);
        assert!(text.contains("<|im_start|>assistant\n<think>"));
    }

    fn olmo() -> OlmoInstruct {
        let tok = make_tok(&[
            "<|im_start|>",
            "<|im_end|>",
            "\n",
            "system",
            "Hello",
            "user",
            "assistant",
            "environment",
            "<|endoftext|>",
            "<functions>",
            "</functions>",
            "<function_calls>",
            "</function_calls>",
            "<think>",
            "</think>",
        ]);
        OlmoInstruct::new(tok)
    }

    #[test]
    fn full_conversation() {
        let inst = olmo();
        let mut tokens = Vec::new();
        tokens.extend(inst.system("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.assistant("Hello"));
        tokens.extend(inst.user("Hello"));
        tokens.extend(inst.cue());
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(
            text,
            "<|im_start|>system\nHello<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\nHello<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn answer_uses_environment_role() {
        let inst = olmo();
        let tokens = inst.answer("fn", "Hello");
        let text = inst.tokenizer.decode(&tokens, false);
        assert_eq!(text, "<|im_start|>environment\nHello<|im_end|>\n");
    }

    #[test]
    fn tool_call_grammar_returns_ebnf() {
        let inst = olmo();
        let tools = vec![r#"{"function":{"name":"get_weather","parameters":{}}}"#.to_string()];
        let grammar = inst.tool_call_grammar(&tools);
        assert!(grammar.is_some());
        let g = grammar.unwrap();
        assert!(g.source.contains("root"));
        assert!(g.source.contains("get_weather"));
        assert!(g.source.contains("<function_calls>"));
    }

    #[test]
    fn tool_call_grammar_none_for_empty() {
        let inst = olmo();
        assert!(inst.tool_call_grammar(&[]).is_none());
    }

    #[test]
    fn render_tool_calls_matches_reference_call_syntax() {
        let inst = olmo();
        let calls = vec![(
            "get_weather".to_string(),
            r#"{"city":"Boston","days":3}"#.to_string(),
        )];
        let rendered = inst.render_tool_calls(&calls);
        assert_eq!(
            rendered,
            r#"<function_calls>get_weather(city="Boston", days=3)</function_calls>"#
        );
    }

    #[test]
    fn render_tool_calls_joins_multiple_with_single_wrapper() {
        let inst = olmo();
        let calls = vec![
            ("a".to_string(), "{}".to_string()),
            ("b".to_string(), "{}".to_string()),
        ];
        let rendered = inst.render_tool_calls(&calls);
        assert_eq!(rendered, "<function_calls>a()\nb()</function_calls>");
        assert_eq!(rendered.matches("<function_calls>").count(), 1);
        assert_eq!(rendered.matches("</function_calls>").count(), 1);
    }

    #[test]
    fn tool_decoder_parses_call_syntax() {
        let v: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "\n",
            "system",
            "user",
            "assistant",
            "environment",
            "<|endoftext|>",
            "<functions>",
            "</functions>",
            "<function_calls>",
            "</function_calls>",
            "<think>",
            "</think>",
            r#"get_weather(city="Boston")"#,
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&v));
        let inst = OlmoInstruct::new(tok);
        let mut dec = inst.tool_decoder();
        dec.feed(&[10]); // <function_calls> → enters Inside, returns Start
        dec.feed(&[14]); // get_weather(city="Boston")
        let event = dec.feed(&[11]); // </function_calls>
        match event {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "get_weather");
                assert_eq!(args, r#"{"city":"Boston"}"#);
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }

    #[test]
    fn tool_decoder_parses_multiple_calls_keeps_first() {
        let v: Vec<String> = vec![
            "<|im_start|>",
            "<|im_end|>",
            "\n",
            "system",
            "user",
            "assistant",
            "environment",
            "<|endoftext|>",
            "<functions>",
            "</functions>",
            "<function_calls>",
            "</function_calls>",
            "<think>",
            "</think>",
            "a()\nb()",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&v));
        let inst = OlmoInstruct::new(tok);
        let mut dec = inst.tool_decoder();
        dec.feed(&[10]); // <function_calls>
        dec.feed(&[14]); // a()\nb()
        let event = dec.feed(&[11]); // </function_calls>
        match event {
            ToolEvent::Call(name, args) => {
                assert_eq!(name, "a");
                assert_eq!(args, "{}");
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }
}
