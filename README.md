<div align="center">
  <h1>🚀 Supercharged Vector Database!</h1>

  <a href="#requirements">Requirements</a>
  &nbsp;&bull;&nbsp;
  <a href="#installation">Installation</a>
  &nbsp;&bull;&nbsp;
  <a href="#using-the-program">Using the Program</a>
  &nbsp;&bull;&nbsp;
  <a href="#request-a-feature-or-report-a-bug">Request a Feature or Report a Bug</a>
  &nbsp;&bull;&nbsp;
  <a href="#contact">Contact</a>
</div>

Create and search a vector database to get a response from the large language model that's more accurate.  This is commonly referred to as "retrieval augmented generation" (RAG)!  You can watch an introductory [Video](https://www.youtube.com/watch?v=8-ZAYI4MvtA) or read a [Medium article](https://medium.com/@vici0549/search-images-with-vector-database-retrieval-augmented-generation-rag-3d5a48881de5) about the program. <br>

<details><summary>Graphic of How This Program Works</summary>
  
![image](https://github.com/user-attachments/assets/b3784da7-91a5-426b-882c-756468ffdc20)

</details>

<div align="center">
  <h3><u>Requirements</u></h3>

| [🐍 Python 3.11](https://www.python.org/downloads/release/python-3119/) or [Python 3.12](https://www.python.org/downloads/release/python-3129/) &nbsp;&bull;&nbsp; [📁 Git](https://git-scm.com/downloads) &nbsp;&bull;&nbsp; [📁 Git LFS](https://git-lfs.com/) &nbsp;&bull;&nbsp; [🌐 Pandoc](https://github.com/jgm/pandoc/releases) &nbsp;&bull;&nbsp; [🛠️ Compiler](https://visualstudio.microsoft.com/) |
|---|

The above link downloads Visual Studio as an example.  Make sure to install the required SDKs, however.
 
> <details>
>   <summary>EXAMPLE error when no compiler installed:</summary>
>   <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/sample_error.png?raw=true">
> </details>
> 
> <details>
>   <summary>EXAMPLE of installing the correct SDKs:</summary>
>   <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/build_tools.png?raw=true">
> </details>

</div>

[Back to Top](#top)

<a name="installation"></a>
<div align="center"> <h2>Installation</h2></div>
  
### Step 1
Go to the latest "release" and download the ZIP file.  Extract its contents and navigate to the `src` folder.
  * NOTE: If you simply clone this repository you will get the development version, which may or may not be stable.
### Step 2
Within the `src` folder, create a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/):
```
python -m venv .
```
### Step 3
Activate the virtual environment:
```
.\Scripts\activate
```
### Step 4
Run the setup script:
   > Only ```Windows``` is supported for now.

```
python setup_windows.py
```

[Back to Top](#top)

<a name="using-the-program"></a>
<div align="center"> <h2>🖥️Usage🖥️</h2></div>

### 🔥Important🔥
* More detailed instructions are being consolidated into the `Ask Jeeves` functionality, which can be accessed from the "Ask Jeeves" menu option.  Please create an issue if Jeeves is not working.

### Start the Program
```
.\Scripts\activate
```
```
python gui.py
```

### 🏗️ Create a Vector Database
* Download a vector/embedding model from the ```Models Tab```.
* Select one or more files to be put into the `queue`.
  * 📄 Non-audio files (including images) can be selected by clicking the `Choose Files` button within the `Create Database Tab`.
  * 🎵 Audio files must be transcribed first within the `Tools Tab`.
  * 🖼️ Images can be selected just like other general file types.
* Click the `Create Vector Database` button within the `Create Database Tab`.
  > Make sure and change the `chunk size` and `chunk overlap` settings within the `Settings Tab` first.

### 🔍 Query a Vector Database
* Within the `Query Database Tab`, select the database you want to search.
* Select the backend that will analyze the results from the vector database and provide a response: `Local Models`, `Kobold`, `LM Studio` or `ChatGPT`.
* Click `Submit Question`.
  * The `chunks only` checkbox will display the results from the vector database.

### ❓ Which Backend Should I Use?
If you use either the `Kobold` or `LM Studio` backends you must be familiar with those programs.  For example, `LM Studio` must be running in "server mode" and handles the prompt formatting.  However,`Kobold` automatically starts in server mode but requires you to specify the prompt formatting.  This program no longer provides detailed instructions on how to use either of these two backends.  For more information:
* Kobold [home page](https://github.com/LostRuins/koboldcpp), [instructions](https://github.com/LostRuins/koboldcpp/wiki), and [Discord server](https://koboldai.org/discord)
* LM Studio [home page](https://lmstudio.ai/), [instructions](https://lmstudio.ai/docs), and [Discord server](https://discord.gg/aPQfnNkxGC).

### 🗑️ Deleting a Database
* In the `Manage Databases Tab`, select a database and click `Delete Database`.

[Back to Top](#top)

<a name="request-a-feature-or-report-a-bug"></a>
## Request a Feature or Report a Bug

Feel free to report bugs or request enhancements by creating an issue on github and I will respond promptly.

<a name="contact"></a>
<div align="center"><h3>CONTACT</h3></div>

I welcome all suggestions - both positive and negative.  You can e-mail me directly at "bbc@chintellalaw.com" or I can frequently be seen on the ```KoboldAI``` Discord server (moniker is ```vic49```).  I am always happy to answer any quesitons or discuss anything vector database related!  (no formal affiliation with ```KoboldAI```).

<br>
<div align="center">
    <a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example1.png" target="_blank">
        <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example1.png?raw=true" alt="Example Image" width="350">
    </a>
    <a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example2.png" target="_blank">
        <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example2.png?raw=true" alt="Example Image" width="350">
    </a>
    <a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example3.png" target="_blank">
        <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example3.png?raw=true" alt="Example Image" width="350">
    </a>
    <a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example4.png" target="_blank">
        <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example4.png?raw=true" alt="Example Image" width="350">
    </a>
      </a>
    <a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example5.png" target="_blank">
        <img src="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio/blob/main/src/Assets/example5.png?raw=true" alt="Example Image" width="350">
    </a>
</div>
