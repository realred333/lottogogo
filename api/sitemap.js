module.exports = function sitemap(req, res) {
  const host = req.headers["x-forwarded-host"] || req.headers.host || "localhost";
  const proto = req.headers["x-forwarded-proto"] || "https";
  const baseUrl = `${proto}://${host}`;
  const today = new Date().toISOString().slice(0, 10);
  const pages = [
    { path: "/", changefreq: "weekly", priority: "1.0" },
    { path: "/about.html", changefreq: "monthly", priority: "0.8" },
    { path: "/methodology.html", changefreq: "monthly", priority: "0.8" },
  ];

  const xml =
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' +
    pages
      .map(
        (page) =>
          '  <url>\n' +
          `    <loc>${baseUrl}${page.path}</loc>\n` +
          `    <lastmod>${today}</lastmod>\n` +
          `    <changefreq>${page.changefreq}</changefreq>\n` +
          `    <priority>${page.priority}</priority>\n` +
          '  </url>\n'
      )
      .join("") +
    '</urlset>\n';

  res.setHeader("content-type", "application/xml; charset=utf-8");
  return res.status(200).send(xml);
};
