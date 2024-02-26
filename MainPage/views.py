from typing import Any
from pathlib import Path

from django.shortcuts import render, redirect
from django.views import View

from .forms import ParamtersForm
from .ai_tools import load_all, predict

__all__ = (
    'index',
    'generic',
    'elements',
    'DiseaseChecker'
)


def index(request, disease_chance: float = None):
    return render(request, 'MainPage/index.html', {'disease_chance': disease_chance})


def generic(request):
    return render(request, 'MainPage/generic.html')


def elements(request):
    return render(request, 'MainPage/elements.html')


class DiseaseChecker(View):
    path_model: Path = None
    name: str = None
    _model=None
    _featurespace=None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model, self._featurespace = load_all(path=self.path_model, name=self.name)

    async def get(self, request):
        return redirect('MainPage:index')

    async def post(self, request):
        form = ParamtersForm(request.POST)
        if form.is_valid():
            cleaned = form.cleaned_data
            print(cleaned)
            cleaned['age'] = int(cleaned['age'])
            cleaned['sex'] = int(cleaned['sex'])
            cleaned = {key.replace('_', ' '): value for key, value in cleaned.items()}
            return index(request, disease_chance=predict(
                model=self._model,
                featurespace=self._featurespace,
                parameters=cleaned
            ))
