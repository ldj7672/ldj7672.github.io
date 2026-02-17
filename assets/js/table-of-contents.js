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

    // Wait a bit for content to be fully rendered (especially if markdown processing is async)
    setTimeout(function() {
      // Find all headings (h1, h2, h3, h4, h5, h6) within the content area
      // The page title is outside .page__content, so all headings inside are valid
      const headings = content.querySelectorAll('h1, h2, h3, h4, h5, h6');
      
      // Debug: log found headings
      console.log('Found headings:', headings.length);
      console.log('Content element:', content);
      console.log('Content HTML (first 500 chars):', content.innerHTML.substring(0, 500));
      
      const h1s = content.querySelectorAll('h1');
      console.log('H1 headings found:', h1s.length);
      h1s.forEach((h, i) => {
        console.log(`H1 ${i}: ${h.textContent.substring(0, 50)}`);
      });
      
      headings.forEach((h, i) => {
        console.log(`Heading ${i}: ${h.tagName} - ${h.textContent.substring(0, 50)}`);
      });
      
      if (headings.length === 0) {
        console.log('No headings found in .page__content');
        return;
      }
      
      createTOC(headings, content, article);
    }, 100);
  }

  function createTOC(headings, content, article) {

    // Create TOC container
    const tocContainer = document.createElement('div');
    tocContainer.id = 'table-of-contents';
    tocContainer.className = 'toc-container';

    const tocTitle = document.createElement('div');
    tocTitle.className = 'toc-title';
    tocTitle.innerHTML = '<span>목차</span><button class="toc-toggle" aria-label="목차 닫기">×</button>';
    tocContainer.appendChild(tocTitle);
    
    // Toggle button functionality
    const toggleBtn = tocTitle.querySelector('.toc-toggle');
    let isCollapsed = false;
    toggleBtn.addEventListener('click', function() {
      isCollapsed = !isCollapsed;
      tocContainer.classList.toggle('collapsed', isCollapsed);
      toggleBtn.textContent = isCollapsed ? '☰' : '×';
    });

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
          // Position TOC to the right of sidebar with some padding
          tocContainer.style.left = (sidebarWidth + 20) + 'px';
        } else {
          // If no sidebar, position at left edge
          tocContainer.style.left = '20px';
        }
      }
      
      // Position on load and resize
      positionTOC();
      window.addEventListener('resize', positionTOC);
      
      // Also check on scroll to ensure TOC doesn't overlap sidebar
      window.addEventListener('scroll', function() {
        positionTOC();
      });
    }

    // Highlight active section on scroll and hide/show TOC at top
    function updateActiveTOC() {
      const scrollPos = window.scrollY || window.pageYOffset || document.documentElement.scrollTop;
      const scrollThreshold = 100; // Hide TOC when scrolled less than 100px from top
      
      // Hide TOC when at top of page
      if (scrollPos < scrollThreshold) {
        tocContainer.classList.add('at-top');
      } else {
        tocContainer.classList.remove('at-top');
      }

      const scrollPosWithOffset = scrollPos + 100; // Offset for fixed header

      let current = '';
      headings.forEach((heading) => {
        const headingTop = heading.getBoundingClientRect().top + window.pageYOffset;
        if (scrollPosWithOffset >= headingTop) {
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

  // Original initTOC function wrapper
  function initTOCWrapper() {
    initTOC();
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTOCWrapper);
  } else {
    initTOCWrapper();
  }
  
  // Also try after a delay to catch any late-rendered content
  setTimeout(initTOCWrapper, 500);
})();
