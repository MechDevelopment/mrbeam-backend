from aiohttp import web
import asyncio
import logging
import os
import time


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
        image = post.get("image")
        im = image.file.read()
        task = Task(image=im)

        try:
            await self.flow.process(task, timeout_sec=60)
        except RuntimeError as e:
            if str(e).startswith('CUDA out of memory'):
                logging.error('Internal error: {}'.format(e))
                os.system('kill -9 %d' % os.getpid())
            logging.error('Internal error: {}'.format(e))
            return web.Response(body={'status': str(e)}, status=500)
        
        return web.json_response([{"xmin": 1}, {"xmin": 2}])

def prepare_app() -> web.Application:
    app = web.Application(client_max_size=0)
    app.router.add_post('/inference', PredictView)

    AppIntegrator(app).add_flow(get_flow())

    return app


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    loop = asyncio.get_event_loop()
    web.run_app(prepare_app(), loop=loop, port=8011)
