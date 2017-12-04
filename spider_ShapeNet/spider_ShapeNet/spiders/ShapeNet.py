import scrapy


class ShapeNetSpider(scrapy.Spider):
    name = "ShapeNet"

    def start_requests(self):
        urls = [
            "https://www.shapenet.org/"
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = '{}.html'.format(page)
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)
