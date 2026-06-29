"""FastAPI app exposing the annotation service over REST.

Same engine as the Qt app, different front: a web client fetches ``/api/cloud``
and ``/api/state``, POSTs commands, and re-renders from the returned semantic
state. The built web UI (if present) is served at ``/``.
"""

from __future__ import annotations

import importlib.resources as resources

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from toaster.core import LabelSchema

from .service import AnnotationService

__all__ = ["create_app"]


class OpenBody(BaseModel):
    path: str


class PickBody(BaseModel):
    index: int
    modifiers: list[str] = []


class BoxBody(BaseModel):
    indices: list[int]
    modifiers: list[str] = []


class AssignBody(BaseModel):
    class_id: int | None = None


class ActiveClassBody(BaseModel):
    class_id: int


class DisplayModeBody(BaseModel):
    mode: str


class SegmentBody(BaseModel):
    name: str
    params: dict = {}
    scope_to_selection: bool = True


class GroupSelectBody(BaseModel):
    group_id: int
    modifiers: list[str] = []


class GroupAssignBody(BaseModel):
    group_id: int
    class_id: int | None = None


class SuggestedBody(BaseModel):
    group_id: int | None = None


class VisibilityBody(BaseModel):
    group_id: int
    visible: bool


class SoloBody(BaseModel):
    group_id: int


class AddClassBody(BaseModel):
    name: str
    color: list[int] | str | None = None


class RenameClassBody(BaseModel):
    class_id: int
    name: str


class ClassColorBody(BaseModel):
    class_id: int
    color: list[int] | str


class RemoveClassBody(BaseModel):
    class_id: int


def create_app(schema: LabelSchema | None = None) -> FastAPI:
    """Build the FastAPI app around a single :class:`AnnotationService`."""
    app = FastAPI(title="Toaster", version="0.1.0")
    service = AnnotationService(schema)
    app.state.service = service

    @app.exception_handler(RuntimeError)
    async def _runtime_error(_request, exc: RuntimeError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error(_request, exc: ValueError):
        # e.g. invalid segmenter parameters — a client mistake, not a server bug.
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # -- reads --
    @app.get("/api/meta")
    def meta():
        return service.meta()

    @app.get("/api/cloud")
    def cloud():
        return service.cloud()

    @app.get("/api/state")
    def state():
        return service.state()

    # -- lifecycle / selection / annotation --
    @app.post("/api/open")
    def open_cloud(body: OpenBody):
        return service.open_cloud(body.path)

    @app.post("/api/pick")
    def pick(body: PickBody):
        return service.pick(body.index, body.modifiers)

    @app.post("/api/box")
    def box(body: BoxBody):
        return service.box(body.indices, body.modifiers)

    @app.post("/api/assign")
    def assign(body: AssignBody):
        return service.assign(body.class_id)

    @app.post("/api/active_class")
    def active_class(body: ActiveClassBody):
        return service.set_active_class(body.class_id)

    @app.post("/api/display_mode")
    def display_mode(body: DisplayModeBody):
        return service.set_display_mode(body.mode)

    @app.post("/api/undo")
    def undo():
        return service.undo()

    @app.post("/api/redo")
    def redo():
        return service.redo()

    @app.post("/api/clear_selection")
    def clear_selection():
        return service.clear_selection()

    @app.post("/api/save")
    def save():
        return service.save()

    # -- segmentation / groups --
    @app.post("/api/segment")
    def segment(body: SegmentBody):
        return service.run_segmenter(body.name, body.params, body.scope_to_selection)

    @app.post("/api/group/select")
    def group_select(body: GroupSelectBody):
        return service.select_group(body.group_id, body.modifiers)

    @app.post("/api/group/assign")
    def group_assign(body: GroupAssignBody):
        return service.assign_group(body.group_id, body.class_id)

    @app.post("/api/group/suggested")
    def group_suggested(body: SuggestedBody):
        return service.apply_suggested(body.group_id)

    @app.post("/api/group/visibility")
    def group_visibility(body: VisibilityBody):
        return service.set_group_visibility(body.group_id, body.visible)

    @app.post("/api/group/solo")
    def group_solo(body: SoloBody):
        return service.solo_group(body.group_id)

    @app.post("/api/groups/show_all")
    def groups_show_all():
        return service.show_all_groups()

    # -- class (schema) editing --
    @app.post("/api/class/add")
    def class_add(body: AddClassBody):
        return service.add_class(body.name, body.color)

    @app.post("/api/class/rename")
    def class_rename(body: RenameClassBody):
        return service.rename_class(body.class_id, body.name)

    @app.post("/api/class/color")
    def class_color(body: ClassColorBody):
        return service.set_class_color(body.class_id, body.color)

    @app.post("/api/class/remove")
    def class_remove(body: RemoveClassBody):
        return service.remove_class(body.class_id)

    # -- static web front (served at / when built) --
    web_dir = resources.files("toaster") / "web"
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app
