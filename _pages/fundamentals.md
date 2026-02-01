---
title: "Fundamentals"
permalink: /fundamentals/
layout: archive
author_profile: true
sidebar:
  nav: sidebar_nav
---

{% include base_path %}

{% assign posts = site.fundamentals | sort: 'date' | reverse %}
{% for post in posts %}
  {% include archive-single.html %}
{% endfor %}
