from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from mrds.core.config import get_settings

settings = get_settings()

connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_async_engine(
    settings.DATABASE_URL, echo=False, future=True, connect_args=connect_args
)

async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)
