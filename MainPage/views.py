from django.shortcuts import render, redirect


from .forms import ParamtersForm


__all__ = (
    'index',
    'generic',
    'elements',
    'check_disease'
)


def index(request,
          disease_chance: float = None
          ):
    return render(request, 'MainPage/index.html', 
                  {'disease_chance': disease_chance}
                  )


def generic(request):
    return render(request, 'MainPage/generic.html')


def elements(request):
    return render(request, 'MainPage/elements.html')


def check_disease(request):
    if request.method != 'POST':
        return redirect('MainPage:index')
    form = ParamtersForm(request.POST)
    if form.is_valid():
        return index(request, disease_chance=0.31254)
