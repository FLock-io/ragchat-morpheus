## 🎬 Get Started

### 📝 Prerequisites

- CPU >= 4 cores
- RAM >= 16 GB
- Disk >= 50 GB
- Docker >= 24.0.0 & Docker Compose >= v2.26.1
  > If you have not installed Docker on your local machine (Windows, Mac, or Linux),
  see [Install Docker Engine](https://docs.docker.com/engine/install/).

### 🚀 Start up the server

1. Clone the repo:

   ```bash
   $ git clone https://github.com/FLock-io/ragchat-morpheus.git
   ```

2. Build the pre-built Docker images and start up the server:

   > Running the following commands automatically downloads the *dev* version RAGFlow Docker image. To download and run
   a specified Docker version, update `RAGFLOW_VERSION` in **docker/.env** to the intended version, for
   example `RAGFLOW_VERSION=v0.8.0`, before running the following commands.

   ```bash
   $ cd ragchat-morpheus
   $ docker compose up 
   ```

   > The core image is about 16 GB in size and may take a while to load.

3. Check the server status after having the server up and running:

   _The following output confirms a successful launch of the system:_

   ```bash
    * Running on all addresses (0.0.0.0)
    * Running on http://127.0.0.1:8000
    * Running on http://x.x.x.x:8000
    INFO:werkzeug:Press CTRL+C to quit
   ```
4.  Run your openai script.

```bash
from openai import OpenAI

client = OpenAI(
    api_key="fake-api-key",
    base_url="http://127.0.0.1:8000" # If it's not local, replace it with your url
)
stream = client.chat.completions.create(
    model="morpheus-model",
    messages=[{"role": "user", "content": "what is mor?"}
              ]
)
print(stream.choices[0].message.content)
   ```
It may take a while to download the model on your first visit.


