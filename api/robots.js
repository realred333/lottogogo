module.exports = function robots(req, res) {
  const host = req.headers["x-forwarded-host"] || req.headers.host || "localhost";
  const proto = req.headers["x-forwarded-proto"] || "https";
  const baseUrl = `${proto}://${host}`;

  const body = ["User-agent: *", "Allow: /", `Sitemap: ${baseUrl}/sitemap.xml`, ""].join("\n");

  res.setHeader("content-type", "text/plain; charset=utf-8");
  return res.status(200).send(body);
};
