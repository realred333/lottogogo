module.exports = function sitemap(req, res) {
  const host = req.headers["x-forwarded-host"] || req.headers.host || "localhost";
  const proto = req.headers["x-forwarded-proto"] || "https";
  const baseUrl = `${proto}://${host}`;
  const today = new Date().toISOString().slice(0, 10);

  const xml =
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' +
    '  <url>\n' +
    `    <loc>${baseUrl}/</loc>\n` +
    `    <lastmod>${today}</lastmod>\n` +
    '    <changefreq>weekly</changefreq>\n' +
    '    <priority>1.0</priority>\n' +
    '  </url>\n' +
    '</urlset>\n';

  res.setHeader("content-type", "application/xml; charset=utf-8");
  return res.status(200).send(xml);
};
