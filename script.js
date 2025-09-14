/**
 * BIFRÖST Research Website JavaScript
 * Interactive functionality for the research project website
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeSmoothScrolling();
    initializeCopyToClipboard();
    initializeScrollAnimations();
    initializeMobileMenu();
});

/**
 * Initialize navigation functionality
 */
function initializeNavigation() {
    const header = document.querySelector('.header');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    // Handle scroll effect on header
    let lastScrollTop = 0;
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > 100) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
        
        lastScrollTop = scrollTop;
    });
    
    // Handle active navigation links
    const sections = document.querySelectorAll('section[id]');
    window.addEventListener('scroll', function() {
        const scrollPosition = window.scrollY + 100;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            const navLink = document.querySelector(`.nav-links a[href="#${sectionId}"]`);
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                navLinks.forEach(link => link.classList.remove('active'));
                if (navLink) {
                    navLink.classList.add('active');
                }
            }
        });
    });
}

/**
 * Initialize smooth scrolling for anchor links
 */
function initializeSmoothScrolling() {
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                e.preventDefault();
                
                const offsetTop = targetSection.offsetTop - 80; // Account for fixed header
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                const navToggle = document.querySelector('.nav-toggle');
                const navLinks = document.querySelector('.nav-links');
                if (navToggle.classList.contains('active')) {
                    navToggle.classList.remove('active');
                    navLinks.classList.remove('active');
                }
            }
        });
    });
}

/**
 * Initialize copy to clipboard functionality
 */
function initializeCopyToClipboard() {
    window.copyToClipboard = function(selector) {
        const element = document.querySelector(selector);
        if (element) {
            const text = element.textContent || element.innerText;
            
            // Create temporary textarea element
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            
            // Select and copy text
            textarea.select();
            document.execCommand('copy');
            
            // Remove temporary element
            document.body.removeChild(textarea);
            
            // Show feedback
            showCopyFeedback();
        }
    };
    
    function showCopyFeedback() {
        const copyBtn = document.querySelector('.copy-btn');
        const originalText = copyBtn.textContent;
        
        copyBtn.textContent = '✅ Copied!';
        copyBtn.style.background = '#10b981';
        
        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.background = '';
        }, 2000);
    }
}

/**
 * Initialize scroll animations
 */
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.method-step, .result-item, .code-card, .citation-card');
    animateElements.forEach(el => {
        observer.observe(el);
    });
}

/**
 * Initialize mobile menu functionality
 */
function initializeMobileMenu() {
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function() {
            this.classList.toggle('active');
            navLinks.classList.toggle('active');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            }
        });
    }
}

/**
 * Utility function to check if element is in viewport
 */
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

/**
 * Initialize lazy loading for images
 */
function initializeLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver(function(entries, observer) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
}

/**
 * Enhanced scroll effects
 */
function initializeScrollEffects() {
    let ticking = false;
    
    function updateScrollEffects() {
        const scrollTop = window.pageYOffset;
        const hero = document.querySelector('.hero');
        
        if (hero) {
            // Parallax effect for hero section
            const parallaxOffset = scrollTop * 0.3;
            hero.style.transform = `translateY(${parallaxOffset}px)`;
        }
        
        ticking = false;
    }
    
    window.addEventListener('scroll', function() {
        if (!ticking) {
            requestAnimationFrame(updateScrollEffects);
            ticking = true;
        }
    });
}

/**
 * Initialize search functionality (if needed)
 */
function initializeSearch() {
    const searchInput = document.querySelector('.search-input');
    const searchResults = document.querySelector('.search-results');
    
    if (searchInput && searchResults) {
        let searchTimeout;
        
        searchInput.addEventListener('input', function() {
            const query = this.value.trim();
            
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                if (query.length > 2) {
                    performSearch(query);
                } else {
                    searchResults.innerHTML = '';
                    searchResults.style.display = 'none';
                }
            }, 300);
        });
    }
}

/**
 * Performance optimization: Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Add CSS for additional animations and mobile menu
 */
const additionalStyles = `
/* Mobile menu styles */
@media (max-width: 768px) {
    .nav-toggle.active span:nth-child(1) {
        transform: rotate(45deg) translate(5px, 5px);
    }
    
    .nav-toggle.active span:nth-child(2) {
        opacity: 0;
    }
    
    .nav-toggle.active span:nth-child(3) {
        transform: rotate(-45deg) translate(7px, -6px);
    }
    
    .nav-links {
        position: fixed;
        top: 70px;
        left: 0;
        width: 100%;
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        flex-direction: column;
        padding: 2rem;
        transform: translateX(-100%);
        transition: transform 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .nav-links.active {
        transform: translateX(0);
    }
    
    .nav-links a {
        padding: 1rem 0;
        border-bottom: 1px solid var(--border-color);
        text-align: center;
        font-size: 1.1rem;
    }
    
    .nav-links a:last-child {
        border-bottom: none;
    }
}

/* Animation classes */
.animate-in {
    animation: fadeInUp 0.6s ease-out forwards;
}

/* Scroll header effect */
.header.scrolled {
    background: rgba(255, 255, 255, 0.98);
    box-shadow: var(--shadow-light);
}

/* Active navigation link */
.nav-links a.active {
    color: var(--primary-color);
}

.nav-links a.active::after {
    width: 100%;
}

/* Lazy loading */
img.lazy {
    opacity: 0;
    transition: opacity 0.3s;
}

img.lazy.loaded {
    opacity: 1;
}
`;

// Add the additional styles to the document
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);