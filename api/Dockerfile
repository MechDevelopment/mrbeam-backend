FROM rust:1.68.2 AS builder
WORKDIR /app
RUN apt update && apt install lld clang -y
COPY . .
ENV SQLX_OFFLINE true
RUN cargo build --release

FROM debian:buster-slim AS runtime
WORKDIR /app
RUN apt-get update
RUN apt-get install -y libssl-dev
COPY --from=builder /app/target/release/api api
COPY .env.docker .env