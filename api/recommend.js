const TIMEOUT_MS = 45000;

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
    const contentType = upstream.headers.get("content-type") || "application/json; charset=utf-8";

    res.status(upstream.status);
    res.setHeader("content-type", contentType);
    res.setHeader("cache-control", "no-store");
    return res.send(payloadText);
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
