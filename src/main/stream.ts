/**
 * Consuming an AI text stream, shared by 截图模式's answers (shortcuts.ts) and
 * 对话模式's hints (conversation.ts), which each map the outcome to their own
 * IPC events.
 */

/**
 * One slice of a model's reply. Thinking models reason before they answer;
 * `reasoning` deltas carry that reasoning so the renderer can show it while
 * it forms. `text` is the answer itself and the only part kept in the
 * conversation history.
 */
export type StreamChunk = { kind: 'reasoning'; delta: string } | { kind: 'text'; delta: string }

export type StreamOutcome =
  /** Ran to the end on its own */
  | { status: 'complete'; text: string; reasoning: string }
  /** Cut short through the controller; the caller knows why */
  | { status: 'aborted'; text: string; reasoning: string }
  | { status: 'failed'; text: string; reasoning: string; error: unknown }

/**
 * Forward every chunk until the stream ends, fails or is aborted. Chunks still
 * arriving after an abort are dropped, so a request that replaced this one
 * never sees them.
 */
export async function consumeStream(
  createStream: (signal: AbortSignal) => AsyncIterable<StreamChunk>,
  controller: AbortController,
  onChunk: (chunk: StreamChunk) => void
): Promise<StreamOutcome> {
  const { signal } = controller
  let text = ''
  let reasoning = ''
  try {
    for await (const chunk of createStream(signal)) {
      if (signal.aborted) break
      if (chunk.kind === 'reasoning') reasoning += chunk.delta
      else text += chunk.delta
      onChunk(chunk)
    }
  } catch (error) {
    // An abort surfaces as an AbortError; it is not a failure
    if (!signal.aborted) return { status: 'failed', text, reasoning, error }
  }
  return { status: signal.aborted ? 'aborted' : 'complete', text, reasoning }
}

type ApiError = Error & {
  responseBody?: string
  statusCode?: number
}

/**
 * Extract meaningful error message from API errors. A request longer than the
 * model takes is explained, since 资料库 material makes that easy to reach.
 */
export function extractErrorMessage(error: unknown): string {
  const message = platformErrorMessage(error)
  if (!isContextOverflow(error)) return message
  return `请求内容超出了模型能处理的长度，请到「设置 → 资料库」少用一些资料，或换一个上下文更长的模型（${message}）`
}

function platformErrorMessage(error: unknown): string {
  if (!(error instanceof Error)) {
    return String(error) || '未知错误'
  }

  // Try to extract responseBody from AI SDK errors
  const apiError = error as ApiError

  // Try to parse responseBody for detailed message
  if (apiError.responseBody) {
    try {
      const body = JSON.parse(apiError.responseBody)
      if (body.message) {
        return body.message
      }
      if (body.error?.message) {
        return body.error.message
      }
    } catch {
      // If parsing fails, use responseBody as is
      if (typeof apiError.responseBody === 'string' && apiError.responseBody.length < 200) {
        return apiError.responseBody
      }
    }
  }

  // Fallback to error message
  return error.message || '未知错误'
}

/**
 * How platforms word "this model takes no images": OpenAI (`image_url is only
 * supported by certain models`), DeepSeek (`unknown variant \`image_url\``),
 * vLLM (`is not a multimodal model`), OpenRouter (`support image input`)…
 * Kept narrow on purpose: an image that is merely too large must not count.
 */
const IMAGE_REFUSAL =
  /image_url is only supported|unknown variant `image_url`|not a multi-?modal model|(?:does not|doesn't|not) supports? (?:image|vision)|support image input|不支持(?:图片|图像|多模态|视觉)/i

/** Whether the request failed because the model does not take image input */
export function isImageInputRefused(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  const { statusCode, responseBody, message } = error as ApiError
  if (statusCode !== undefined && statusCode !== 400 && statusCode !== 422) return false
  return IMAGE_REFUSAL.test(`${responseBody ?? ''}\n${message}`)
}

/**
 * How platforms word "the request is longer than the model takes": OpenAI and
 * vLLM (`maximum context length`, `context_length_exceeded`), DashScope
 * (`Range of input length should be`), others (`prompt is too long`, `too many
 * tokens`)
 */
const CONTEXT_OVERFLOW =
  /context[_ ]length|maximum context|context window|prompt is too long|too many tokens|input length|(?:超出|超过)[^，。]{0,8}(?:上下文|长度)/i

function isContextOverflow(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  const { statusCode, responseBody, message } = error as ApiError
  if (statusCode !== undefined && statusCode !== 400 && statusCode !== 413) return false
  return CONTEXT_OVERFLOW.test(`${responseBody ?? ''}\n${message}`)
}
