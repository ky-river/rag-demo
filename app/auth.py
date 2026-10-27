from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class User:
    id: str
    email: str


def normalize_email(email: str) -> str:
    normalized = email.strip().casefold()
    if len(normalized) > 254 or not EMAIL_PATTERN.fullmatch(normalized):
        raise AuthError("请输入有效邮箱")
    return normalized


def validate_password(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise AuthError(f"密码至少需要 {PASSWORD_MIN_LENGTH} 位")
    if len(password) > PASSWORD_MAX_LENGTH:
        raise AuthError(f"密码不能超过 {PASSWORD_MAX_LENGTH} 位")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=32,
    )
    return (
        f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}$"
        f"{salt.hex()}${digest.hex()}"
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_hex, digest_hex = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(bytes.fromhex(digest_hex)),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init_db(self) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                "id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, "
                "password_hash TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS auth_sessions ("
                "token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
                "expires_at INTEGER NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_sessions_user "
                "ON auth_sessions(user_id)"
            )

    def register(self, email: str, password: str) -> tuple[User, str]:
        normalized_email = normalize_email(email)
        validate_password(password)
        user = User(id=uuid4().hex, email=normalized_email)
        try:
            with closing(self.connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO users(id, email, password_hash) VALUES (?, ?, ?)",
                    (user.id, user.email, hash_password(password)),
                )
        except sqlite3.IntegrityError as exc:
            raise AuthError("该邮箱已经注册，请直接登录", 409) from exc
        return user, self.create_session(user.id)

    def login(self, email: str, password: str) -> tuple[User, str]:
        normalized_email = normalize_email(email)
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ?",
                (normalized_email,),
            ).fetchone()
        if not row or not verify_password(password, row[2]):
            raise AuthError("邮箱或密码错误", 401)
        user = User(id=row[0], email=row[1])
        return user, self.create_session(user.id)

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + TOKEN_TTL_SECONDS
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ?",
                (int(time.time()),),
            )
            connection.execute(
                "INSERT INTO auth_sessions(token_hash, user_id, expires_at) "
                "VALUES (?, ?, ?)",
                (hash_token(token), user_id, expires_at),
            )
        return token

    def authenticate(self, token: str) -> User | None:
        if not token:
            return None
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT users.id, users.email "
                "FROM auth_sessions "
                "JOIN users ON users.id = auth_sessions.user_id "
                "WHERE auth_sessions.token_hash = ? "
                "AND auth_sessions.expires_at > ?",
                (hash_token(token), int(time.time())),
            ).fetchone()
        return User(id=row[0], email=row[1]) if row else None

    def revoke(self, token: str) -> None:
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (hash_token(token),),
            )
