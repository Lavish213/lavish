# inside fetch_async()
async def fetch_async(self):
    try:
        resp = await self.session.get(self.url)
        if resp.status == 429:
            raise Exception("429 rate-limit")
        return await resp.json()
    except Exception as e:
        raise e