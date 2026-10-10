(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    initReadingProgress();
    initCodeBlocks();
    initScrollSpy();
    initBackToTop();
  });

  // --------------------------------------------------------------------------
  // 1. Reading Progress Bar
  // --------------------------------------------------------------------------
  function initReadingProgress() {
    const progressBar = document.getElementById('reading-progress-bar');
    const article = document.querySelector('.post-article');

    if (!progressBar || !article) return;

    let ticking = false;

    function updateProgress() {
      const articleRect = article.getBoundingClientRect();
      const articleTop = window.scrollY + articleRect.top;
      const articleHeight = article.offsetHeight;
      const windowHeight = window.innerHeight;

      // Start calculating when top of article reaches near top of screen
      const totalScrollable = articleHeight - windowHeight + 100;
      const currentScroll = window.scrollY - articleTop;

      let percent = 0;
      if (totalScrollable > 0) {
        percent = (currentScroll / totalScrollable) * 100;
        percent = Math.max(0, Math.min(100, percent));
      } else if (window.scrollY > articleTop) {
        percent = 100;
      }

      progressBar.style.width = percent + '%';
      ticking = false;
    }

    window.addEventListener('scroll', function () {
      if (!ticking) {
        window.requestAnimationFrame(updateProgress);
        ticking = true;
      }
    }, { passive: true });

    window.addEventListener('resize', updateProgress);
    updateProgress();
  }

  // --------------------------------------------------------------------------
  // 2. Code Block Enhancement (Language Badge + Copy Button)
  // --------------------------------------------------------------------------
  function initCodeBlocks() {
    const codeBlocks = document.querySelectorAll('.post-content .highlight, .post-content pre:not(.chroma)');

    codeBlocks.forEach(function (block) {
      // Check if already processed
      if (block.closest('.code-block-wrapper')) return;

      const pre = block.matches('pre') ? block : block.querySelector('pre');
      if (!pre) return;

      const code = pre.querySelector('code');
      let lang = 'CODE';

      if (code) {
        if (code.getAttribute('data-lang')) {
          lang = code.getAttribute('data-lang');
        } else {
          const classList = Array.from(code.classList);
          const langClass = classList.find(c => c.startsWith('language-'));
          if (langClass) {
            lang = langClass.replace('language-', '');
          }
        }
      }

      // Friendly language formatting
      lang = lang.toUpperCase().trim();
      if (!lang || lang === '') lang = 'CODE';

      // Create container
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      // Create header
      const header = document.createElement('div');
      header.className = 'code-header';

      const langSpan = document.createElement('span');
      langSpan.className = 'code-lang';
      langSpan.textContent = lang;

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.className = 'code-copy-btn';
      copyBtn.setAttribute('aria-label', 'Copy code to clipboard');
      copyBtn.innerHTML = '<i class="fa-regular fa-copy" aria-hidden="true"></i> <span>Copy</span>';

      copyBtn.addEventListener('click', function () {
        const textToCopy = code ? code.innerText : pre.innerText;
        if (!navigator.clipboard) {
          // Fallback
          const textarea = document.createElement('textarea');
          textarea.value = textToCopy;
          document.body.appendChild(textarea);
          textarea.select();
          try {
            document.execCommand('copy');
            showCopied(copyBtn);
          } catch (e) {}
          document.body.removeChild(textarea);
          return;
        }

        navigator.clipboard.writeText(textToCopy).then(function () {
          showCopied(copyBtn);
        }).catch(function () {});
      });

      header.appendChild(langSpan);
      header.appendChild(copyBtn);

      // Insert wrapper into DOM
      const parent = block.parentNode;
      parent.insertBefore(wrapper, block);
      wrapper.appendChild(header);
      wrapper.appendChild(block);
    });

    function showCopied(btn) {
      const originalHTML = btn.innerHTML;
      btn.classList.add('copied');
      btn.innerHTML = '<i class="fa-solid fa-check" aria-hidden="true"></i> <span>Copied!</span>';
      setTimeout(function () {
        btn.classList.remove('copied');
        btn.innerHTML = originalHTML;
      }, 2000);
    }
  }

  // --------------------------------------------------------------------------
  // 3. Active Section ScrollSpy for Table of Contents
  // --------------------------------------------------------------------------
  function initScrollSpy() {
    const headings = document.querySelectorAll('.post-content h2[id], .post-content h3[id]');
    const sidebarLinks = document.querySelectorAll('#sidebar-toc a[href^="#"]');
    const inlineLinks = document.querySelectorAll('#post-toc-inline a[href^="#"]');

    if (headings.length === 0 || sidebarLinks.length === 0) return;

    let activeHeadingId = null;

    function getActiveHeading() {
      let current = null;

      for (let i = 0; i < headings.length; i++) {
        const heading = headings[i];
        const rect = heading.getBoundingClientRect();
        if (rect.top <= 140) {
          current = heading.id;
        } else {
          break;
        }
      }

      // If at the top of the article, highlight the first heading if visible
      if (!current && headings.length > 0) {
        const firstRect = headings[0].getBoundingClientRect();
        if (firstRect.top < window.innerHeight * 0.5) {
          current = headings[0].id;
        }
      }

      return current;
    }

    function updateActiveTOC() {
      const currentId = getActiveHeading();
      if (!currentId || currentId === activeHeadingId) return;

      activeHeadingId = currentId;

      // Update sidebar links
      sidebarLinks.forEach(function (link) {
        const href = link.getAttribute('href');
        if (href === '#' + currentId) {
          link.classList.add('active');
          // Auto-scroll sidebar if link is out of view in sidebar
          scrollSidebarLinkIntoView(link);
        } else {
          link.classList.remove('active');
        }
      });

      // Update inline links
      inlineLinks.forEach(function (link) {
        const href = link.getAttribute('href');
        if (href === '#' + currentId) {
          link.classList.add('active');
        } else {
          link.classList.remove('active');
        }
      });
    }

    function scrollSidebarLinkIntoView(link) {
      const sidebar = document.querySelector('.sidebar-toc');
      if (!sidebar) return;

      const sidebarRect = sidebar.getBoundingClientRect();
      const linkRect = link.getBoundingClientRect();

      if (linkRect.top < sidebarRect.top || linkRect.bottom > sidebarRect.bottom) {
        link.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }

    let ticking = false;
    window.addEventListener('scroll', function () {
      if (!ticking) {
        window.requestAnimationFrame(function () {
          updateActiveTOC();
          ticking = false;
        });
        ticking = true;
      }
    }, { passive: true });

    updateActiveTOC();
  }

  // --------------------------------------------------------------------------
  // 4. Back to Top Button
  // --------------------------------------------------------------------------
  function initBackToTop() {
    const btn = document.getElementById('btn-back-to-top');
    if (!btn) return;

    btn.addEventListener('click', function (e) {
      e.preventDefault();
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }

})();
