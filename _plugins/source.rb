class HelloWorld < Liquid::Tag

  def initialize(tag_name, text, tokens)
    super
    @text = text
  end

  def render(context)
    "#{@text}"
  end

end

Liquid::Template.register_tag('source', HelloWorld)