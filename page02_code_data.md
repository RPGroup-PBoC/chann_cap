---
layout: page
title: Code and Data
img: code.png # Add image post (optional)
permalink: code
sidebar: true
---

---

## The `chann_cap` Module
This work required several home-made Python functions. To ensure
reproducibility, we have written it as a Python module that can be installed
from the master branch of the [GitHub
repository](http://wwww.github.com/rpgroup-pboc/chann_cap). Please see the
installation instructions for details. This module is required to execute all
of the following scripts.

{% if site.data.figures %}
## Main Text Figure Generation

This section contains all of the scripts necessary along with the required
datasets to reproduce all plots from the main text. Click on the preview images
to obtain a PDF version of the figure, or click on the script title to download
the *.py* file to reproduce the figure. 

{% for fig in site.data.figures %}
<article class="post">

<a class="post-thumbnail" style="background-image: url({{site.url}}/{{site.baseurl}}/assets/img/{{fig.pic}})" href="{{site.baseurl}}/assets/pdf/{{fig.pdf}}"> </a>

<div class="post-content">
<b class="post-title"><a href="{{site.url}}/{{site.baseurl}}/software/figs/{{fig.filename}}">{{fig.title}}</a></b>
<p> {{fig.desc}}</p>

<i>Necessary Data Sets </i><br/>
{% for ds in fig.req %}
  {% if ds.storage == 'local' %}
    {% assign link = "{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}" %}
  {% else %}
    {% assign link = "{{ds.link}}" %}
  {% endif %}
  <span>&#8226;</span> <a style="font-size: 0.9em;" href="{{link}}"> {{ds.title}} </a><br/>
{% endfor %}
</div>
</article>
{% endfor %}
{% endif %}

{% if site.data.code %}
## Jupyter Notebooks

This section contains detailed code in the format of Jupyter notebooks. These
notebooks extensively explain the logic behind the computations that went into
each of the sections with highly annotated Markdown text. The notebooks can be
viewed as *html* files or can be downloaded as *ipynb* to be executed. When
necessary, there is a link to download the data used for the computations in
the notebook.

{% for script in site.data.code %}
* [**{{script.name}}**]({{site.url}}/{{site.baseurl}}/software/{{script.name}}.html)
  \| [[ipynb file]]({{site.url}}/{{site.baseurl}}/software/{{script.name}}.ipynb)
  + {{script.desc}}
  {% if script.req %} 
    <i>Necessary Data Sets </i><br/>
    {% for ds in script.req %}
      {% if ds.storage == 'local' %}
        {% assign link = "{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}" %}
      {% else %}
        {% assign link = "{{ds.link}}" %}
      {% endif %}
    <span>&#8226;</span> <a style="font-size: 0.9em;" href="{{link}}"> {{ds.title}} </a><br/>
    {% endfor %}
  {% endif %}
{% endfor %}
{% endif %}

{% if site.data.sifigures%}
## Supplemental Information Figure Generation

This section contains the scripts and links to the necessary datasets to
generate all figures in the supplemental materials.

{% for sifig in site.data.sifigures %}
* [**{{sifig.title}}**]({{site.url}}/{{site.baseurl}}/software/figs/{{sifig.filename}})
  + {{sifig.desc}}
  {% if sifig.req %} 
    <i>Necessary Data Sets </i><br/>
    {% for ds in sifig.req %}
      {% if ds.storage == 'local' %}
        {% assign link = "{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}" %}
      {% else %}
        {% assign link = "{{ds.link}}" %}
      {% endif %}
    <span>&#8226;</span> <a style="font-size: 0.9em;" href="{{link}}"> {{ds.title}} </a><br/>
    {% endfor %}
  {% endif %}
{% endfor %}
{% endif %}

{% if site.data.scripts %}
## Python scripts
This section lists python scripts used to compute repetitive tasks explained in
the Jupyter notebooks. When necessary, there is a link to download the data
used for the computations in the notebook.

{% for script in site.data.scripts %}
* [**{{script.name}}**]({{site.url}}/{{site.baseurl}}/software/scripts/{{script.name}})
  {% if script.dataset %} \| [[data]]({{script.dataset}}){% endif %}
    + {{script.desc}}
{% endfor %}
{% endif %}

{% if site.data.datasets %}
## Data Sets

This section lists all datasets used for this work. From the raw microscopy
images, to the processed single-cell fluorescence values. Also here we list all
values generated from theoretical calculations that are computationally
expensive to reproduce every single time.

{% for ds in site.data.datasets %}
* [{{ds.name}}]({%if ds.storage !=
  'remote'%}{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}{%
  else%}{{site.link}}{% endif %}) \| {% if ds.filetype %}(filetype:
  {{ds.filetype}}){%endif%}{% if ds.filesize %}({{ds.filesize}}){%endif%}{%
  if ds.storage.remote %} DOI: {{ds.DOI}}{%endif%}
{% endfor %}
{% endif %}