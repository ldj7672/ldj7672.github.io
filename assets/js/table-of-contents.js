/**
 * Table of Contents Generator
 * Automatically generates a table of contents from headings in the page content
 */
(function() {
  'use strict';

  function initTOC() {
    // Only run on single post pages
    const article = document.querySelector('article.page');
    if (!article) return;

    const content = document.querySelector('.page__content');
    if (!content) return;

    // Find all headings (h2, h3, h4, h5, h6)
    const headings = content.querySelectorAll('h2, h3, h4, h5, h6');
    if (headings.length === 0) return;

    // Create TOC container
    const tocContainer = document.createElement('div');
    tocContainer.id = 'table-of-contents';
    tocContainer.className = 'toc-container';

    const tocTitle = document.createElement('div');
    tocTitle.className = 'toc-title';
    tocTitle.textContent = '목차';
    tocContainer.appendChild(tocTitle);

    const tocList = document.createElement('ul');
    tocList.className = 'toc-list';

    // Generate IDs for headings if they don't have them
    headings.forEach((heading, index) => {
      if (!heading.id) {
        // Create ID from heading text
        heading.id = 'heading-' + index + '-' + heading.textContent
          .toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-')
          .replace(/-+/g, '-')
          .trim();
      }

      // Create TOC item
      const tocItem = document.createElement('li');
      tocItem.className = 'toc-item toc-level-' + heading.tagName.toLowerCase();

      const tocLink = document.createElement('a');
      tocLink.href = '#' + heading.id;
      tocLink.textContent = heading.textContent;
      tocLink.className = 'toc-link';

      // Smooth scroll
      tocLink.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.getElementById(heading.id);
        if (target) {
          const offset = 80; // Offset for fixed header
          const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - offset;
          window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
          });
          // Update URL without jumping
          history.pushState(null, null, '#' + heading.id);
        }
      });

      tocItem.appendChild(tocLink);
      tocList.appendChild(tocItem);
    });

    tocContainer.appendChild(tocList);

    // Insert TOC before the article
    const main = document.getElementById('main');
    if (main) {
      main.insertBefore(tocContainer, article);
      // Add class to main for CSS styling
      main.classList.add('has-toc');
      
      // Calculate sidebar width and position TOC accordingly
      function positionTOC() {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
          const sidebarRect = sidebar.getBoundingClientRect();
          const sidebarWidth = sidebarRect.width || 250; // Default to 250px if not found
          tocContainer.style.left = sidebarWidth + 'px';
        } else {
          // If no sidebar, position at left edge
          tocContainer.style.left = '0px';
        }
      }
      
      // Position on load and resize
      positionTOC();
      window.addEventListener('resize', positionTOC);
    }

    // Highlight active section on scroll
    function updateActiveTOC() {
      const scrollPos = window.scrollY + 100; // Offset for fixed header

      let current = '';
      headings.forEach((heading) => {
        const headingTop = heading.getBoundingClientRect().top + window.pageYOffset;
        if (scrollPos >= headingTop) {
          current = heading.id;
        }
      });

      // Update active state
      tocList.querySelectorAll('.toc-link').forEach((link) => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + current) {
          link.classList.add('active');
        }
      });
    }

    // Throttle scroll events
    let ticking = false;
    window.addEventListener('scroll', function() {
      if (!ticking) {
        window.requestAnimationFrame(function() {
          updateActiveTOC();
          ticking = false;
        });
        ticking = true;
      }
    });

    // Initial update
    updateActiveTOC();
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTOC);
  } else {
    initTOC();
  }
})();
