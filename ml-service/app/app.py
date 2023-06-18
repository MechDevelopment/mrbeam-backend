import asyncio
import logging
from aiohttp import web

from settings import settings, Settings


from aqueduct.integrations.aiohttp import (
    FLOW_NAME,
    AppIntegrator,
)
from flow import (
    Flow,
    Task,
    get_flow,
)


class PredictView(web.View):
    @property
    def flow(self) -> Flow:
        return self.request.app[FLOW_NAME]

    async def post(self):
        post = await self.request.post()
        image = post.get("file")
        im = image.file.read()
        task = Task(image=im)

        try:
            await self.flow.process(task, timeout_sec=60)
        except RuntimeError as e:
            logging.error('Internal error: {}'.format(e))
            return web.Response(body={'status': str(e)}, status=500)
        
        return web.json_response([{"xmin": 1}, {"xmin": 2}])


def prepare_app(settings: Settings) -> web.Application:
    app = web.Application(client_max_size=0)
    app.router.add_post('/predict', PredictView)

    AppIntegrator(app).add_flow(get_flow(settings))

    return app


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    loop = asyncio.get_event_loop()
    web.run_app(prepare_app(settings), loop=loop, port=settings.port)
