---
title: "AI Projects"
permalink: /ai-projects/
layout: archive
author_profile: true
sidebar:
  nav: sidebar_nav
---

{% include base_path %}

{% assign posts = site.ai_projects | sort: 'date' | reverse %}
{% for post in posts %}
  {% include archive-single.html %}
{% endfor %}