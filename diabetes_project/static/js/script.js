document.getElementById('predict-form').addEventListener('submit', async function(e){
  e.preventDefault();
  const form = e.target;
  const data = {
    Pregnancies: form.Pregnancies.value,
    Glucose: form.Glucose.value,
    BloodPressure: form.BloodPressure.value,
    SkinThickness: form.SkinThickness.value,
    Insulin: form.Insulin.value,
    BMI: form.BMI.value,
    DiabetesPedigreeFunction: form.DiabetesPedigreeFunction.value,
    Age: form.Age.value,
  };

  // Basic client-side validation
  for (const [k,v] of Object.entries(data)){
    if (v === '' || isNaN(Number(v)) || Number(v) < 0){
      showResult('Please provide valid non-negative numbers for all fields.', true);
      return;
    }
  }

  showResult('Predicting...', false);

  try{
    const res = await fetch('/api/predict/', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(data),
    });
    if (!res.ok){
      const txt = await res.text();
      showResult('Error: ' + txt, true);
      return;
    }
    const body = await res.json();
    let html = `<strong>Prediction:</strong> ${body.prediction}`;
    if (body.confidence !== null && body.confidence !== undefined){
      html += `<br><strong>Confidence:</strong> ${body.confidence}%`;
    }
    showResult(html, false);
  }catch(err){
    showResult('Network error: ' + err.message, true);
  }
});

function showResult(msg, isError){
  const el = document.getElementById('result');
  el.innerHTML = msg;
  el.style.background = isError ? '#fdecea' : '#eaf3ff';
  el.style.color = isError ? '#7a0b0b' : '#03396b';
}
