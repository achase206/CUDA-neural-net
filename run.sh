#!/bin/sh

BASE_IMAGE=$(cat .docker/image_name)
IMAGE=$(cat .docker/build_image_name)
PORT="${1:-0}"

# Pull the published base image if missing (PyCUDA etc.; RDKit added in child build).
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    echo "Base image not found locally. Pulling $BASE_IMAGE..."
    if ! docker pull "$BASE_IMAGE"; then
        echo "Failed to pull base image $BASE_IMAGE." >&2
        exit 1
    fi
    echo ""
    echo ""
    echo ""
fi

# Build a local image on top of the base (repo mounted at /work at runtime).
echo "Building $IMAGE from $BASE_IMAGE..."
if ! docker build -f .docker/Dockerfile \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "$IMAGE" \
    . ; then
    echo "Failed to build $IMAGE." >&2
    exit 1
fi

# Copy the run script from the image
CID=$(docker create "$IMAGE")
docker cp "$CID:/interface.sh" .interface.sh > /dev/null
docker rm -v "$CID" > /dev/null

# Run the image's interface script
if [ "$PORT" -eq 0 ]; then
    bash .interface.sh "$IMAGE"
else
    bash .interface.sh "$IMAGE" "$PORT"
fi
