up:
	docker compose up --build

down:
	docker compose down

restart:
	docker compose down
	docker compose up --build

logs:
	docker compose logs -f

clean:
	docker compose down -v
	docker system prune -f

reset:
	docker compose down -v
	docker system prune -a -f

test-gpu:
	docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi