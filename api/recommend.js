const TIMEOUT_MS = 45000;
const JSON_CONTENT_TYPE = "application/json; charset=utf-8";

function isJsonContentType(contentType) {
  return typeof contentType === "string" && contentType.toLowerCase().includes("application/json");
}

function summarizePayload(text) {
  if (typeof text !== "string") return "";

  const trimmed = text.trim();
  if (!trimmed) return "";

  if (/^<!doctype html/i.test(trimmed) || /^<html/i.test(trimmed)) {
    return "백엔드가 HTML 오류 페이지를 반환했습니다.";
  }

  return trimmed.replace(/\s+/g, " ").slice(0, 160);
}

module.exports = async function recommendProxy(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ detail: "Method Not Allowed" });
  }

  const baseUrl = (process.env.RENDER_BACKEND_URL || "").trim().replace(/\/$/, "");
  if (!baseUrl) {
    return res.status(500).json({ detail: "RENDER_BACKEND_URL is not configured" });
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const upstream = await fetch(`${baseUrl}/api/recommend`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req.body ?? {}),
      signal: controller.signal,
    });

    clearTimeout(timer);

    const payloadText = await upstream.text();
    const contentType = upstream.headers.get("content-type") || "";
    const isJson = isJsonContentType(contentType);

    res.setHeader("cache-control", "no-store");

    if (isJson) {
      res.status(upstream.status);
      res.setHeader("content-type", contentType || JSON_CONTENT_TYPE);
      return res.send(payloadText);
    }

    const detail = summarizePayload(payloadText);

    if (upstream.ok) {
      return res.status(502).json({
        detail: detail
          ? `백엔드 응답 형식 오류: ${detail}`
          : "백엔드 응답 형식 오류: JSON 응답이 필요합니다.",
      });
    }

    return res.status(upstream.status).json({
      detail: detail ? `백엔드 오류 응답(${upstream.status}): ${detail}` : `백엔드 오류 응답(${upstream.status})`,
    });
  } catch (error) {
    clearTimeout(timer);

    if (error && error.name === "AbortError") {
      return res
        .status(504)
        .json({ detail: "백엔드 준비가 지연되고 있습니다. 잠시 후 다시 시도해주세요." });
    }

    return res.status(502).json({
      detail: error instanceof Error ? `백엔드 호출 실패: ${error.message}` : "백엔드 호출 실패",
    });
  }
};
