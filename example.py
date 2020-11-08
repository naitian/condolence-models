# from condolence_models.condolence_classifier import CondolenceClassifier
from condolence_models.empathy_classifier import EmpathyClassifier

# cc = CondolenceClassifier()
# 
# print("I like ice cream")
# print(cc.predict("I like ice cream"))
# 
# print(["I'm so sorry for your loss.", "F", "Tuesday is a good day of the week."])
# print(cc.predict(["I'm so sorry for your loss.", "F", "Tuesday is a good day of the week."]))

ec = EmpathyClassifier(use_cuda=True, cuda_device=2)

print([["", "Yes, but wouldn't that block the screen?"]])
print(ec.predict([["", "Yes, but wouldn't that block the screen?"]]))
