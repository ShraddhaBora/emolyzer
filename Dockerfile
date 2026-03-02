# stage 1: build react frontend
FROM node:20-slim AS build-stage
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# stage 2: build python backend
FROM python:3.10-slim

# Create a non-root user (Hugging Face expects UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# copy everything else
COPY --chown=user . .

# copy the built frontend from stage 1
COPY --from=build-stage --chown=user /app/frontend/dist /app/frontend/dist

# hf spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# start the app
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
