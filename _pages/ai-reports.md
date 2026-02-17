---
title: "AI Reports"
permalink: /ai-reports/
layout: archive
author_profile: true
sidebar:
  nav: sidebar_nav
---

{% include base_path %}

{% comment %} 카테고리별로 최신 글 3개씩 표시 {% endcomment %}

{% comment %} MLLM 카테고리 {% endcomment %}
{% assign mllm_posts = site.ai_reports | where: "category", "mllm" | sort: 'date' | reverse %}
{% if mllm_posts.size > 0 %}
  <div style="margin-top: 2em; margin-bottom: 1.5em; padding-bottom: 0.5em; border-bottom: 2px solid #e8e8e8;">
    <h2 style="margin: 0; font-size: 1.5em;">
      <a href="{{ '/ai-reports/mllm/' | relative_url }}" style="color: #0066cc; text-decoration: none; font-weight: 600;">MLLM</a>
    </h2>
  </div>
  {% for post in mllm_posts limit: 3 %}
    {% include archive-single.html %}
  {% endfor %}
{% endif %}

{% comment %} Perception 카테고리 {% endcomment %}
{% assign perception_posts = site.ai_reports | where: "category", "perception" | sort: 'date' | reverse %}
{% if perception_posts.size > 0 %}
  <div style="margin-top: 2em; margin-bottom: 1.5em; padding-bottom: 0.5em; border-bottom: 2px solid #e8e8e8;">
    <h2 style="margin: 0; font-size: 1.5em;">
      <a href="{{ '/ai-reports/perception/' | relative_url }}" style="color: #0066cc; text-decoration: none; font-weight: 600;">Perception</a>
    </h2>
  </div>
  {% for post in perception_posts limit: 3 %}
    {% include archive-single.html %}
  {% endfor %}
{% endif %}

{% comment %} Image & Video Generation 카테고리 {% endcomment %}
{% assign image_video_posts = site.ai_reports | where: "category", "image-video-generation" | sort: 'date' | reverse %}
{% if image_video_posts.size > 0 %}
  <div style="margin-top: 2em; margin-bottom: 1.5em; padding-bottom: 0.5em; border-bottom: 2px solid #e8e8e8;">
    <h2 style="margin: 0; font-size: 1.5em;">
      <a href="{{ '/ai-reports/image-video-generation/' | relative_url }}" style="color: #0066cc; text-decoration: none; font-weight: 600;">Image & Video Generation</a>
    </h2>
  </div>
  {% for post in image_video_posts limit: 3 %}
    {% include archive-single.html %}
  {% endfor %}
{% endif %}

{% comment %} Engineering 카테고리 {% endcomment %}
{% assign engineering_posts = site.ai_reports | where: "category", "engineering" | sort: 'date' | reverse %}
{% if engineering_posts.size > 0 %}
  <div style="margin-top: 2em; margin-bottom: 1.5em; padding-bottom: 0.5em; border-bottom: 2px solid #e8e8e8;">
    <h2 style="margin: 0; font-size: 1.5em;">
      <a href="{{ '/ai-reports/engineering/' | relative_url }}" style="color: #0066cc; text-decoration: none; font-weight: 600;">Engineering</a>
    </h2>
  </div>
  {% for post in engineering_posts limit: 3 %}
    {% include archive-single.html %}
  {% endfor %}
{% endif %}