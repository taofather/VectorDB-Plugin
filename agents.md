# Project overview
This file describes this repository, aiming to help the user into developing new features or fix bugs.

## Purpose of the project

A gui interface allowing vectorial search model to query vectorial databases, and submitting context to local models, LMStudio in this case. 

## Structure and purpose of directories

```
src/                        # Main source code.
├── database/               # Factory & implementations for Postgresql with pgvector and TileDB vector databases.
├── setup.py                # Very important file, initializing the application and dependencies.
├── gui.py                  # File which should start the app GUI.

Docs_for_DB/                # Files ingested into vector database are symlinked here.
Models/                     # Downloaded models.
Tokenizer/                  # Tokenizer data.
Dockerfile.postgres         # Dockerfile to create a postgresql image including pgvector extension.
```

## Configuration of the project

### File: config.yaml

The app uses a config.yaml file.
database:
  type: pgvector # means that app will use pgvector implementation.
database:
  type: tiledb # means that app will use tiledb implementation.

### File: docker-compose.yml
Service: postgresql-vector container

Configuration: Uses the same settings from your config.yaml:
- Database: vectordb
- User: postgres
- Password: postgres
- Port: 5433 (mapped from container's 5432)

Features:
- Persistent data storage with named volume
- Health checks to ensure database readiness
- Automatic restart policy
- Network isolation

 ### File: init.sql
Automatically enables the pgvector extension when the database starts
Runs during container initialization

### Script: manage-db.sh
Convenient commands to manage your database:
./manage-db.sh start - Start the database
./manage-db.sh stop - Stop the database gracefully
./manage-db.sh restart - Restart the database
./manage-db.sh status - Check container status
./manage-db.sh logs - View database logs
./manage-db.sh shell - Connect to PostgreSQL shell
./manage-db.sh build - Rebuild the database image
./manage-db.sh clean - Remove everything (with confirmation)

## Postgresql and pgvector

The app connects to a Postgresql instance. Priority is to make it work in postgresql. In my case I use postgres-vector container in docker, pointing to port 5433 (to avoid collisions with another postgres instance in port 5432).

The postgres-vector container was created using the Dockerfile.postgres file.

`docker build -t postgres-vector -f Dockerfile.postgres .`
`docker run --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres-vector`
`docker exec -it postgres-vector psql -U postgres -c "CREATE DATABASE vectordb;"`
docker exec -it postgres-vector psql -U postgres -d vectordb -c "CREATE EXTENSION vector;"


## Source code framework

The src folder contain the main scripts for the app.
- setup.py should run to download and update packages.
- gui.py is the file to run. It triggers a graphic interface with different features.

## GUI features by tab
The GUI offers different tabs to access its functionalities:

### Settings
 * configure LM Studio Host and Port
 * select a DAtabase to query
 * define size and overlap for chunks sent to LM
 * Define TTS Backend, selecting devices available (CPU or GPU) to run.
 * select Vision models, by device.

### Models
* select transformer models to fetch and transform vectorial data and train models.

### Tools
For model training.
* Transcribe audio files.
* Scrape Documentation of models.
* Test vision models.
* Optical character recognition.
* Misc

### Create database
* Choose files to add to a Vector Database (button).
* Select a model to middleware Vector Database.
* Create a Vector Database.

The Choose files button opens a modal allowing:
* Select directory.
* Select files.

### Manage Databases
* Select database.
* View database imported files.
* Delete Database.

### Query Database
* Select a database (select).
* Load the Local Model (select).
* Input text as a model request (textarea).
* Record Audio as a model request (button).
* Transcribe response into audio (button).
* Submit request (button).
* Copy response (button).

## Logging

Logging level should be DEBUG at the time, as only local run is observed nowadays.

```
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Packages use

### Unstructured

`unstructured` is needed to proces markdown files and other kinds like (EPUB, RTF, ODT, Markdown, Email, Excel).



