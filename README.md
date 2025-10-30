# `CIS6930_T7_MTRAG`

`/blue/cis6930/team7/CIS6930_T7_MTRAG`

Simple overview of use/purpose.

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

Create and activate a conda environment (using python 3.11):
```
conda create -p mtraghome python=3.11
conda activate ./mtraghome
```

Install via pip in a conda env for training and inference (tested on python 3.11):
```
pip install llama-cookbook
```

Install the below in a separate conda env for evaluation (tested on python 3.11):
- `torch==2.8.0`
- the `requirements.txt` file in `mt-rag-benchmark/scripts/evaluation`
Note, use as many cores as possible when installing flash-attn. I used 16 cores on hipergator with 32GB ram. More info in the IBM repo's readme.

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
