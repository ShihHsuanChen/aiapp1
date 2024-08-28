import os
import shutil
from typing import List
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import HttpUrl, BaseModel

from . import __version__


class ResultItemSchema(BaseModel):
    label: str
    prob: float


def create_app(cfg):
    from .model import load_model
    from .data import read_image

    model = load_model(cfg.infer_model_name)

    app = FastAPI(
        title='Myaiapp',
        description='Doing image classification using MobileNet',
        version=__version__,
    )

    @app.get('/')
    def root():
        return RedirectResponse('/docs')

    @app.post(
        '/inference/upload',
    )
    def inference_upload(
            file: UploadFile = File(...),
            topk: int = Form(5, ge=1),
            ):
        try:
            suffix = os.path.splitext(file.filename)[-1]
            with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                shutil.copyfileobj(file.file, tmp)
                try:
                    img = read_image(tmp.name)
                except Exception as e:
                    raise HTTPException(
                        status_code=406,
                        detail='Given url is not a valid image file',
                    )
                result = model.inference(img, topk=topk)
        finally:
            file.file.close()
        return [ResultItemSchema(label=label, prob=prob) for label, prob in result]

    @app.post(
        '/inference/url',
        response_model=List[ResultItemSchema],
    )
    def inference_url(
            url: HttpUrl = Form(...),
            topk: int = Form(5, ge=1),
            ):
        try:
            img = read_image(str(url))
        except Exception as e:
            raise HTTPException(
                status_code=406,
                detail='Given url is not a valid image file',
            )
        result = model.inference(img, topk=topk)
        return [ResultItemSchema(label=label, prob=prob) for label, prob in result]
    return app


from .configs import cfg
app = create_app(cfg)
