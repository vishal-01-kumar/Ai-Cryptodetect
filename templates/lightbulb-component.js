// Light Bulb Component JS
function initializeLightBulb(selector) {
    const container = document.querySelector(selector);
    const bulb = container.querySelector('.bulb');
    const cordContainer = container.querySelector('.cord-container');

    let isOn = true;

    function toggleLight() {
        isOn = !isOn;
        
        if (isOn) {
            bulb.classList.add('on');
            container.classList.remove('off');
        } else {
            bulb.classList.remove('on');
            container.classList.add('off');
        }
    }

    cordContainer.addEventListener('click', toggleLight);

    container.addEventListener('mouseover', () => {
        if (isOn) {
            bulb.style.filter = 'brightness(1.1)';
        }
    });

    container.addEventListener('mouseout', () => {
        bulb.style.filter = 'brightness(1)';
    });

    // Initialize default state
    toggleLight();
}
