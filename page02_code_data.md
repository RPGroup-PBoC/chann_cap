---
layout: page
title: Code and Data
img: code.png # Add image post (optional)
permalink: code
sidebar: true
---

---

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
    {% if script.dataset %} [[data]]({{script.dataset}}){% endif %}
    + {{script.desc}}
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
{% for ds in site.data.datasets %}
* [{{ds.name}}]({%if ds.storage !=
  'remote'%}{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}{%
  else%}{{site.link}}{% endif %}) \| {% if ds.filetype %}(filetype:
  {{ds.filetype}}){%endif%}{% if ds.filesize %}({{ds.filesize}}){%endif%}{%
  if ds.storage.remote %} DOI: {{ds.DOI}}{%endif%}
{% endfor %}
{% endif %}

{% if site.data.figures %}
## Figure Generation

{% for fig in site.data.figures %}
<article class="post">

<a class="post-thumbnail" style="background-image: url({{site.url}}/{{site.baseurl}}/assets/img/{{fig.pic}})" href="{{site.baseurl}}/figures/{{fig.pdf}}"> </a>

<div class="post-content">
<b class="post-title"><a href="{{site.url}}/{{site.baseurl}}/software/{{fig.filename}}">{{fig.title}}</a></b>
<p> {{fig.desc}}</p>

<i>Necessary Data Sets </i><br/>
{% for ds in fig.req %}
{% if ds.storage == 'local' %}
{% assign link = "{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}" %}
{% else %}
{% assign link = "{{ds.link}}" %}
{% endif %}
<a style="font-size: 0.9em;" href="{{link}}"> - {{ds.title}} </a><br/>
{% endfor %}
</div>
</article>
{%endfor%}
{% endif %}
