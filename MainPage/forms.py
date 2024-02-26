from django import forms


__all__ = (
    'ParamtersForm',
)


class ParamtersForm(forms.Form):
    # sex = forms.ChoiceField(choices=['1', '2'])
    age = forms.DecimalField(max_value=150)
    IN_T = forms.FloatField(min_value=0)
    IN_P = forms.FloatField(min_value=0)
    IN_P1 = forms.FloatField(min_value=0)
    IN_P2 = forms.FloatField(min_value=0)
    IN_P3 = forms.FloatField(min_value=0)
    OUT_T = forms.FloatField(min_value=0)
    OUT_P = forms.FloatField(min_value=0)
    OUT_P1 = forms.FloatField(min_value=0)
    OUT_P2 = forms.FloatField(min_value=0)
    OUT_P3 = forms.FloatField(min_value=0)
