---
title: "Home"
permalink: /
layout: archive
author_profile: true
sidebar:
  nav: sidebar_nav
---

{% include base_path %}

# 전체 게시글

{% comment %}AI Projects 섹션{% endcomment %}
{% assign ai_projects = site.ai_projects | sort: 'date' | reverse %}
{% if ai_projects.size > 0 %}
<h2>AI Projects</h2>
{% for post in ai_projects %}
  {% include archive-single.html %}
{% endfor %}
{% endif %}

{% comment %}AI Reports 섹션{% endcomment %}
{% assign ai_reports = site.ai_reports | sort: 'date' | reverse %}
{% if ai_reports.size > 0 %}
<h2>AI Reports</h2>
{% for post in ai_reports %}
  {% include archive-single.html %}
{% endfor %}
{% endif %}
