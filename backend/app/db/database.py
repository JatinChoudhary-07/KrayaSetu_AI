from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./dispatcher.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DispatcherOverride(Base):
    __tablename__ = "dispatcher_overrides"

    id = Column(Integer, primary_key=True, index=True)
    work_id = Column(String, index=True, unique=True)
    override_type = Column(String)  # FORCE_INCLUDE, PIN_TIME
    parameters = Column(JSON)       # e.g., {"start_time": "...", "end_time": "..."}

Base.metadata.create_all(bind=engine)
