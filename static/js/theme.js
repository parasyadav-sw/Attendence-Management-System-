// Theme Management
function getTheme(){return localStorage.getItem('theme')||(window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light')}
function setTheme(t){document.documentElement.setAttribute('data-theme',t);localStorage.setItem('theme',t);updateThemeIcon(t)}
function toggleTheme(){setTheme(getTheme()==='dark'?'light':'dark')}
function updateThemeIcon(t){
    document.querySelectorAll('.theme-toggle-sidebar span').forEach(function(s){
        s.textContent=t==='dark'?'Light Mode':'Dark Mode'
    });
    document.querySelectorAll('.theme-toggle-sidebar i').forEach(function(i){
        i.setAttribute('data-lucide',t==='dark'?'sun':'moon')
    });
    if(typeof lucide!=='undefined')lucide.createIcons()
}
document.addEventListener('DOMContentLoaded',function(){setTheme(getTheme());if(typeof lucide!=='undefined')lucide.createIcons()});

// Sidebar Toggle
function toggleSidebar(){document.getElementById('sidebar').classList.toggle('open')}
function closeSidebar(){document.getElementById('sidebar').classList.remove('open')}

// Flash Messages Auto-dismiss
document.addEventListener('DOMContentLoaded',function(){
    document.querySelectorAll('.flash').forEach(function(flash){
        setTimeout(function(){flash.style.animation='slideOut .3s ease forwards';setTimeout(function(){flash.remove()},300)},4000)
    })
});

// Ripple Effect
document.addEventListener('click',function(e){
    const btn=e.target.closest('.btn');
    if(!btn)return;
    const circle=document.createElement('span');
    circle.style.cssText='position:absolute;width:100px;height:100px;background:rgba(255,255,255,.2);border-radius:50%;transform:scale(0);pointer-events:none;';
    const rect=btn.getBoundingClientRect();
    circle.style.left=(e.clientX-rect.left-50)+'px';
    circle.style.top=(e.clientY-rect.top-50)+'px';
    btn.appendChild(circle);
    circle.animate([{transform:'scale(0)',opacity:1},{transform:'scale(4)',opacity:0}],{duration:600});
    setTimeout(function(){circle.remove()},600)
});
