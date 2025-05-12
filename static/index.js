function previewImage(input) {
  const preview = document.getElementById('preview');
  const previewImg = document.getElementById('previewImg');

  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = function(e) {
      previewImg.src = e.target.result;
      preview.style.display = 'block';
    }
    reader.readAsDataURL(input.files[0]);
  } else {
    preview.style.display = 'none';
  }
}
