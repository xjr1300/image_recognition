lint:
	pflake8 image_recognition
	isort --check image_recognition
	black --check image_recognition

format:
	isort image_recognition
	black image_recognition

type-check:
	mypy image_recognition
