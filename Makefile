setup:
	docker-compose up
gen:
	python producer/server.py
recon:
	python reconstructor/server.py
clean:
	docker-compose down