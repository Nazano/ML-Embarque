from os import error, name
import docker
from pathlib import Path


def deploy_and_run(image_name="sentiment"):
    client = docker.from_env()
    print("Deploying model in docker container...")

    try:
        image = client.images.get(image_name)
    except docker.errors.ImageNoteFound:
        image = None

    if image:
        print("Killing current running model")
        containers = client.containers.list(all=True)
        for c in containers:         
            if image == c.image:
                c.kill()
                c.remove()

        client.images.remove(image_name)
        
    
    client.images.build(path=str(Path(__file__).parent.joinpath(Path('docker'))), tag=image_name)
    running_container = client.containers.run(image=image_name, ports={'5000/tcp' : 5000}, stderr=True, detach=True)
    print(f"Container status: {running_container.status}")
    return running_container