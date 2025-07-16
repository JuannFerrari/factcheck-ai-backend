from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy import text
from pgvector.sqlalchemy import Vector
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


# Create async engine
def get_database_url() -> str:
    """Get database URL from settings"""
    if settings.neon_database_url:
        # Convert to asyncpg driver
        url = settings.neon_database_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        return url

    # Construct URL from individual components
    if not all(
        [
            settings.neon_database_host,
            settings.neon_database_user,
            settings.neon_database_password,
        ]
    ):
        raise ValueError(
            "Database configuration incomplete. Please set NEON_DATABASE_URL or "
            "all individual database settings."
        )

    return (
        f"postgresql+asyncpg://{settings.neon_database_user}:"
        f"{settings.neon_database_password}@{settings.neon_database_host}:"
        f"{settings.neon_database_port}/{settings.neon_database_name}"
    )


# Initialize engine as None, will be created when needed
engine = None
AsyncSessionLocal = None


def _get_engine():
    """Get or create the database engine"""
    global engine, AsyncSessionLocal

    if engine is None:
        try:
            engine = create_async_engine(
                get_database_url(),
                echo=settings.environment == "development",
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                # Handle Neon SSL requirements
                connect_args={"server_settings": {"jit": "off"}},
            )
            AsyncSessionLocal = async_sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )
        except Exception as e:
            logger.warning(f"Database engine creation failed: {e}")
            # For testing environments, create a mock engine
            if settings.environment == "test" or (
                settings.neon_database_url
                and "test" in settings.neon_database_url.lower()
            ):
                logger.info("Using mock database for testing")
                engine = None
                AsyncSessionLocal = None
            else:
                raise

    return engine, AsyncSessionLocal


# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for models
Base = declarative_base()


class FactCheckRecordDB(Base):
    """Database model for fact-check records with vector support"""

    __tablename__ = "fact_check_records"

    id = Column(Integer, primary_key=True, index=True)
    claim = Column(Text, nullable=False, index=True)
    claim_embedding = Column(
        "claim_embedding", Vector(384), nullable=True
    )  # Store embedding as vector
    verdict = Column(String(20), nullable=False, index=True)
    confidence = Column(Integer, nullable=False)
    reasoning = Column(Text, nullable=False)
    tldr = Column(Text, nullable=True)
    sources_json = Column(Text, nullable=False)  # JSON string of sources
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Index for vector similarity search
    __table_args__ = (
        Index("idx_verdict_confidence", "verdict", "confidence"),
        Index("idx_created_at", "created_at"),
    )


async def init_database():
    """Initialize database tables and extensions"""
    engine, _ = _get_engine()

    if engine is None:
        logger.info("Skipping database initialization - using mock database")
        return

    try:
        async with engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension enabled")

            # Create tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")

            # Create vector index if it doesn't exist
            await conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_claim_embedding_vector
                ON fact_check_records
                USING ivfflat (claim_embedding vector_cosine_ops)
                WITH (lists = 100)
            """
                )
            )
            logger.info("Vector index created")

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def get_db_session():
    """Get database session - async generator for dependency injection"""
    _, session_factory = _get_engine()

    if session_factory is None:
        logger.warning("No database session available - using mock session")
        yield None
        return

    async with session_factory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()


async def close_database():
    """Close database connections"""
    engine, _ = _get_engine()

    if engine is not None:
        await engine.dispose()
        logger.info("Database connections closed")
    else:
        logger.info("No database connections to close")
