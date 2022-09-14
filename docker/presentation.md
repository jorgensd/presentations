---
marp: true
class: lead
paginate: true
math: katex
theme: uncover
style: |
  section {

  background-color: #ccc;
  letter-spacing: 1px;
  text-align: left;

  }
  h1 {

  font-size: 1.3em;
  text-align: center;

  }
  h2 {

  font-size: 1.5em;
  text-align: left;

  }
  h3 {

  font-size: 1em;

  text-align: center;
  font-weight: normal;
  letter-spacing: 1px;

  }
  h6 {

  text-align: center;
  font-weight: normal;
  letter-spacing: 1px;

  }
  p{

  text-align: left;
  font-size: 0.75em;
  letter-spacing: 0px;

  }
  img[src$="centerme"] {
  font-size: 0.8em; 
  display:block; 
  margin: 0 auto; 
  }
  footer{

  color: black;
  text-align: left;

  }
  ul {

  padding: 10;
  margin: 0;

  }
  ul li {

  color: black;
  margin: 5px;
  font-size: 30px;

  }
  /* Code */
  pre, code, tt {

  font-size: 0.98em;
  font-size: 25px;
  font-family: Consolas, Courier, Monospace;
  color: white;

  }
  code , tt{

  margin: 0px;
  padding: 2px;
  white-space: nowrap;
  border: 1px solid #eaeaea;
  border-radius: 3px;

  }

  pre {

  background-color: #f8f8f8;
  overflow: auto;
  padding: 6px 10px;
  border-radius: 3px;
  background-color: black;
  }

  pre code, pre tt {

  background-color: transparent;
  border: none;
  margin: 0;
  padding: 0;
  white-space: pre;
  border: none;
  background: transparent;
  }
---

# Docker

### Jørgen S. Dokken


###### dokken@simula.no

---
# Docker creates a virtual operating system

- [Docker Desktop](https://docs.docker.com/desktop/)
- [Docker Client](https://docs.docker.com/engine/reference/commandline/cli)
- [Docker Hub](https://hub.docker.com/)

```python
docker pull ubuntu:22.04
```
---
# Docker consists of different abstractions

- Docker container images (Custom filesystem with all preinstalled dependencies)
- Docker container (runnable instance of an image on any OS)

---