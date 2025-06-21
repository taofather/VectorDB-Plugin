
# Llegiu-me

## Arquitectura

```mermaidjs
graph TD
    H[Document Chunks] --> I[Embedding Model]
    I --> J[Document Vectors]
    J --> K[Stored in PostgreSQL]

    A[User Query: Aliento de Salamandra] --> B[Embedding Model]
    B --> C[Query Vector]
    C --> D[PostgreSQL Vector Search]
    D --> E[Retrieved Contexts]
    E --> F[LM Studio Model]
    F --> G[Final Response]

```

## Abstraient la bbdd vectioral: afegint suport per Postgres i pgsql

Ara tenim una estructura que suporta tant TileDB com pgvector. Per utilitzar-la:
Quan s'executa setup.py, l'usuari pot triar entre TileDB i pgvector
La configuració es guarda a config.yaml
El codi utilitza el factory per crear la implementació correcta de la base de dades
Per provar-ho:
Executa python setup.py i tria l'opció 2 per pgvector
Assegura't que PostgreSQL està executant-se i que pgvector està instal·lat
Configura les credencials de PostgreSQL a config.yaml si és necessari
Executa l'aplicació normalment amb python gui.py


##Changes done

Abstracted database:
Modificar l'estructura de directoris per suportar ambdues bases de dades.
Modificar la configuració per suportar ambdues opcions a través de config.yaml.

I've made several improvements to the setup script:

1. Removed the dependency on flash-attn for macOS users since it requires CUDA
2. Simplified the package installation process by using a fixed list of priority packages
3. Added better error handling and more informative messages
4. Fixed the directory structure creation to use the correct paths
5. Improved the configuration file handling
6. Made the script more robust by checking for existing configurations

## How to run

Now try running `python setup.py` again. The script should:

1. Skip the flash-attn installation on macOS
2. Install all other required packages
3. Set up the directory structure
4. Configure the system to use pgvector

After the setup completes, you'll need to:
1. Make sure PostgreSQL is running on port 5432
2. Create a database named 'vectordb' in PostgreSQL
3. Install the pgvector extension in your PostgreSQL database

## Problema: segments de PDF desconnectats a Postgres-vector

La segmentació dels documents PDF no és òptima, ja que està trencant seccions que haurien d'estar juntes. Per exemple, en el primer fragment, la descripció de l'encantament "Aliento de Salamandra" està tallada abruptament.

El projecte inicial usaba:
chunk_size: 700 caràcters
chunk_overlap: 250 caràcters

Aquests valors són massa petits per mantenir la coherència semàntica del contingut, especialment per a documents estructurats com els PDFs amb seccions ben definides.

### Solució Ràpida: Augmentar els valors de chunk_size i chunk_overlap
Això millora immediatament la coherència, especialment per documents estructurats:

database:
  chunk_overlap: 400    # abans 250
  chunk_size:    1500   # abans 700



## Testejant l'embedding model

La idea és que aquesta query ens doni una història a partir de la preparació d'un conjur, que és molt simple, ja que només es necessita una gota de sang de salamandra. L'encanteri simplement crea foc, tirant la gota de sang de l'animal sobre material combustible, com si d'un foc normal es tractés.

`Crea un petit Plot per una història de rol  d'Aquelarre on l'objectiu sigui aconseguir els elements per fer el conjur Aliento de Salamandra per escalfar-se en una nit freda d'hivern, donat que els jugadors no disposen d'escaquer ni pedernal.`

En successives proves, el model inventa l'encanteri, posant-hi escates de drac, plomes de Fènix, etc. La idea és que sigui capaç de construïr amb la recepta de l'encanteri que l'embedding model ha guardat a Postgresql.