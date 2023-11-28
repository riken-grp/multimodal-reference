import fcntl
import json
from collections.abc import Collection
from pathlib import Path
from typing import Generic, Optional, TypeVar

ResourceIDType = TypeVar("ResourceIDType")


class FileBasedResourceManagerMixin(Generic[ResourceIDType]):
    def __init__(
        self, available_resources: Collection[ResourceIDType], state_file_path: Path, state_prefix: str = "resource"
    ) -> None:
        self.available_resources = available_resources
        self.state_file_path = state_file_path
        self.state_prefix = state_prefix
        if not self.state_file_path.exists():
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_file_path.write_text(json.dumps({}))

    def acquire_resource(self) -> Optional[ResourceIDType]:
        with open(self.state_file_path, mode="r+") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Exclusive lock for both reading and writing
            state = json.load(file)

            for resource_id in self.available_resources:
                if not state.get(f"{self.state_prefix}_{resource_id}", False):
                    # resource is available, mark as in use
                    state[f"{self.state_prefix}_{resource_id}"] = True
                    file.seek(0)
                    file.truncate()
                    json.dump(state, file)
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                    return resource_id

            # If no GPU is available, release the lock and return None
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            return None

    def release_resource(self, resource_id: ResourceIDType) -> None:
        with open(self.state_file_path, mode="r+") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            state = json.load(file)
            state[f"{self.state_prefix}_{resource_id}"] = False
            file.seek(0)
            file.truncate()
            json.dump(state, file)
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
