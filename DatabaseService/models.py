import sqlalchemy as _sql
import datetime as _dt
from DatabaseService import database as _database


class Contact(_database.Base):
    __tablename__ = 'contacts'
    id = _sql.Column(_sql.Integer, primary_key=True, index=True)
    name = _sql.Column(_sql.String, index=True)
    last_name = _sql.Column(_sql.String, index=True)
    email = _sql.Column(_sql.String, index=True, unique=True)
    phone = _sql.Column(_sql.String, index=True, unique=True)
    date = _sql.Column(_sql.String, index=True)


