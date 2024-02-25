from django.shortcuts import render


__all__ = (
    'index',
    'generic',
    'elements'
)


def index(request):
    return render(request, 'MainPage/index.html')


def generic(request):
    return render(request, 'MainPage/generic.html')


def elements(request):
    return render(request, 'MainPage/elements.html')
