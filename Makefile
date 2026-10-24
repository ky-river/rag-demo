.PHONY: test up down health logs

test:
	python -m unittest discover -s tests -v

up:
	docker compose up --build -d

down:
	docker compose down

health:
	curl -sS http://127.0.0.1:8000/health

logs:
	docker compose logs --tail=100 backend
