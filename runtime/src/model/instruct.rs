//! Instruct trait — model-specific conversational AI formatting and decoding.
//!
//! Each model architecture provides its own implementation. The API layer
//! delegates to the model's `Instruct` impl for all instruct operations.

use crate::inference::structured::grammar::Grammar;
use crate::model::tokenizer::Tokenizer;
use std::sync::Arc;

/// A model-provided tool-call grammar: the EBNF source (used as a stable
/// cache key when compiling for a tokenizer) paired with the parsed AST.
pub struct ToolGrammar {
    pub source: String,
    pub grammar: Arc<Grammar>,
}

// Shared decoders
pub mod decoders;

// Model implementations
pub mod gemma2;
pub mod gemma3;
pub mod gemma4;
pub mod gptoss;
pub mod kimi;
pub mod llama2;
pub mod llama3;
pub mod mistral3;
pub mod olmo2;
pub mod olmo3;
pub mod phi3;
pub mod qwen2;
pub mod qwen3;
pub mod r1;

/// Events emitted by the chat decoder.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Generated text chunk
    Delta(String),
    /// Special token encountered (token ID)
    Interrupt(u32),
    /// Generation complete (full accumulated text)
    Done(String),
}

/// Events emitted by the reasoning decoder.
#[derive(Debug, Clone)]
pub enum ReasoningEvent {
    /// Reasoning block started
    Start,
    /// Reasoning text chunk
    Delta(String),
    /// Reasoning complete (full reasoning text)
    Complete(String),
}

/// Events emitted by the tool decoder.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// Tool call detected
    Start,
    /// Complete tool call: (name, arguments-json)
    Call(String, String),
}

/// Classifies generated tokens into text deltas, interrupts, and done.
pub trait ChatDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ChatEvent;
    fn reset(&mut self);
}

/// Detects reasoning/thinking blocks in the token stream.
pub trait ReasoningDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ReasoningEvent;
    fn reset(&mut self);
}

/// Detects tool call blocks in the token stream.
pub trait ToolDecoder: Send {
    fn feed(&mut self, tokens: &[u32]) -> ToolEvent;
    fn reset(&mut self);
}

/// Model-specific instruct implementation.
///
/// Each architecture provides its own impl with hardcoded tokens & logic.
/// The tokenizer is owned by the implementation to avoid redundant lookups.
pub trait Instruct: Send + Sync {
    fn system(&self, msg: &str) -> Vec<u32>;
    fn first_user(&self, msg: &str) -> Vec<u32> {
        self.user(msg)
    }
    fn user(&self, msg: &str) -> Vec<u32>;
    fn system_user(&self, system: &str, user: &str) -> Vec<u32> {
        let mut tokens = self.system(system);
        tokens.extend(self.user(user));
        tokens
    }
    fn assistant(&self, msg: &str) -> Vec<u32>;
    fn cue(&self) -> Vec<u32>;
    fn seal(&self) -> Vec<u32>;
    fn equip(&self, tools: &[String]) -> Vec<u32>;
    fn answer(&self, name: &str, value: &str) -> Vec<u32>;

    /// Render one assistant turn's tool calls as replay text for this
    /// architecture's chat template, given (name, arguments-json) pairs.
    /// Default reproduces the Hermes/Qwen-style wire format
    /// (`<tool_call>\n{"name":...,"arguments":...}\n</tool_call>`, one per
    /// call, newline-joined) that most chat-apc-supported archs share.
    fn render_tool_calls(&self, calls: &[(String, String)]) -> String {
        calls
            .iter()
            .map(|(name, arguments_json)| {
                let arguments: serde_json::Value = serde_json::from_str(arguments_json)
                    .unwrap_or_else(|_| serde_json::Value::String(arguments_json.clone()));
                format!(
                    "<tool_call>\n{}\n</tool_call>",
                    serde_json::json!({ "name": name, "arguments": arguments })
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder>;
    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder>;
    fn tool_decoder(&self) -> Box<dyn ToolDecoder>;

    /// Returns the parsed tool-call grammar that constrains generation to
    /// the architecture's tool-call format, given a list of tool schemas.
    /// Returns `None` if the architecture doesn't support constrained tool calling.
    fn tool_call_grammar(&self, _tools: &[String]) -> Option<ToolGrammar> {
        None
    }
}

/// Create the appropriate instruct implementation for the given architecture.
pub fn create(arch_name: &str, tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    use self::qwen3::{ChatMLConfig, QwenInstruct};

    match arch_name {
        "qwen3" | "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_moe"
        | "qwen3_vl" | "qwen3_vl_text" => {
            Arc::new(QwenInstruct::new(
                tokenizer,
                ChatMLConfig {
                    has_thinking: true,
                    has_tools: true,
                    generation_suffix: "",
                    stop_tokens: &["<|im_end|>", "<|endoftext|>"],
                },
            ))
        }
        "nemotron_h" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: false,
                generation_suffix: "<think>\n",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
        "qwen2" => Arc::new(self::qwen2::new(tokenizer)),
        "llama2" => Arc::new(self::llama2::LlamaInstruct::new(tokenizer)),
        "llama3" | "l4ma" => Arc::new(self::llama3::LlamaInstruct::new(tokenizer)),
        "r1" | "deepseek_v3" | "deepseek_v4" => Arc::new(self::r1::R1Instruct::new(tokenizer)),
        "kimi_k2" | "kimi_k25" => Arc::new(self::kimi::KimiInstruct::new(tokenizer)),
        "glm_moe_dsa" => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: true,
                has_tools: true,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>", "<|user|>", "<|assistant|>"],
            },
        )),
        "gptoss" | "gpt_oss" => Arc::new(self::gptoss::GptOssInstruct::new(tokenizer)),
        "gemma2" => Arc::new(self::gemma2::GemmaInstruct::new(tokenizer)),
        "gemma3" => Arc::new(self::gemma3::Gemma3Instruct::for_variant(
            tokenizer,
            self::gemma3::Gemma3Variant::Gemma3,
        )),
        "gemma3_text" => Arc::new(self::gemma3::Gemma3Instruct::for_variant(
            tokenizer,
            self::gemma3::Gemma3Variant::Gemma3Text,
        )),
        "gemma3n" => Arc::new(self::gemma3::Gemma3Instruct::for_variant(
            tokenizer,
            self::gemma3::Gemma3Variant::Gemma3n,
        )),
        "gemma3n_text" => Arc::new(self::gemma3::Gemma3Instruct::for_variant(
            tokenizer,
            self::gemma3::Gemma3Variant::Gemma3nText,
        )),
        "gemma4" => Arc::new(self::gemma4::Gemma4Instruct::for_variant(
            tokenizer,
            self::gemma4::Gemma4Variant::Gemma4,
        )),
        "gemma4_text" => Arc::new(self::gemma4::Gemma4Instruct::for_variant(
            tokenizer,
            self::gemma4::Gemma4Variant::Gemma4Text,
        )),
        "mistral3" | "ministral3" => Arc::new(self::mistral3::MistralInstruct::new(tokenizer)),
        "olmo2" => Arc::new(self::olmo2::Olmo2Instruct::new(tokenizer)),
        "olmo3" => Arc::new(self::olmo3::OlmoInstruct::new(tokenizer)),
        "phi3" => Arc::new(self::phi3::Phi3Instruct::new(tokenizer)),
        _ => Arc::new(QwenInstruct::new(
            tokenizer,
            ChatMLConfig {
                has_thinking: false,
                has_tools: false,
                generation_suffix: "",
                stop_tokens: &["<|im_end|>", "<|endoftext|>"],
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `render_tool_calls`'s default body reproduces the Hermes/Qwen-style
    /// wire format chat-apc previously hardcoded directly (before it was
    /// made model-aware). `Olmo2Instruct` doesn't override this method, so
    /// it exercises the trait default. This replaces the coverage the
    /// now-deleted chat-apc test `assistant_tool_calls_replay_as_native_tool_call_payload`
    /// provided — that test called the concrete renderer with no model
    /// seam, which stopped being possible once rendering required a live
    /// `&Model` (see `RatioThink/Inferlets/chat-apc/src/chat/completions.rs`).
    #[test]
    fn render_tool_calls_default_matches_hermes_format() {
        use crate::model::instruct::olmo2::Olmo2Instruct;
        use crate::model::tokenizer::Tokenizer;
        use std::sync::Arc;

        let tok = Arc::new(Tokenizer::from_vocab(&["<|endoftext|>".to_string()]));
        let inst = Olmo2Instruct::new(tok);
        let calls = vec![("calculator".to_string(), r#"{"expr":"2+2"}"#.to_string())];
        let rendered = inst.render_tool_calls(&calls);
        assert!(rendered.contains("<tool_call>"), "{rendered}");
        assert!(rendered.contains("</tool_call>"), "{rendered}");
        assert!(rendered.contains("\"name\":\"calculator\""), "{rendered}");
        assert!(
            rendered.contains("\"arguments\":{\"expr\":\"2+2\"}"),
            "{rendered}"
        );
    }
}
