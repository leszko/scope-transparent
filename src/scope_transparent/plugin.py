"""Plugin class for Daydream Scope integration."""

import logging

from scope.core.plugins import hookimpl

from .pipeline import TransparentPipeline

logger = logging.getLogger(__name__)


class TransparentPlugin:
    """Scope plugin that applies transparency using a mask input."""

    @hookimpl
    def register_pipelines(self, register):
        register(TransparentPipeline)
        logger.info("Registered Transparent pipeline")
